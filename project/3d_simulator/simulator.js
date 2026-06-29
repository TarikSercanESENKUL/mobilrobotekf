/**
 * 3D Autonomous Bus Simulator (Three.js Engine)
 * =============================================
 * Real-time 3D simulation of AKIA EV Bus.
 * Simulates:
 * 1. 3D LiDAR (Rotating scan + point cloud rays)
 * 2. Front 77GHz RADAR (Adaptive Cruise & Headway ACC scan cone)
 * 3. 360° Cameras (6 Transparent FOV Frustums)
 * 4. 12x Ultrasonic sensors (Near field distance arcs)
 * 5. Dual-Antenna RTK GNSS (Skyward beam to satellite)
 * 6. Real-time EKF Ghost Bus (Blue transparent tracker matching EKF output)
 * 7. Dynamic Pedestrian Crossing (t=65-75s) and Traffic Light (s=400m)
 */

// Global Variables
let scene, camera, renderer, controls;
let bus, ghostBus, EKF_UncertaintyDisk;
let trafficLightMesh, pedestrianMesh, roadMesh;
let landmarksGroup, trajectoryLine, ekfLine;

// Simulation variables
let s = 0.0;
let v = 0.0;
let t = 0.0;
let isPaused = false;
let cameraMode = 'chase'; // 'chase', 'top', 'orbit', 'cabin'

// Route parameters (Matching Python RouteGenerator)
const straightLen = 400.0;
const radius = 50.0;
const semiCircleLen = Math.PI * radius;
const lapLen = 2 * straightLen + 2 * semiCircleLen;
const busStops = [200.0, 600.0, 950.0];

// Urban Actors State
let tlColor = "green";
let pedActive = false;
let currentDwell = "CRUISE";

// Sensor activation switches
let activeSensors = {
    lidar: true,
    radar: true,
    camera: true,
    ultrasonic: true,
    rtk: true
};

// Sensor Visual Objects
let lidarBeamLines = [];
let radarConeMesh;
let cameraFrustums = [];
let ultrasonicArcs = [];
let gnssSatelliteLine;

// Path histories
let truePathPoints = [];
let ekfPathPoints = [];

// Real-time EKF States
let ekfX = 0, ekfY = 0, ekfTheta = 0, ekfV = 0;
let ekfError = 0.02;

// ─── 1. KİNEMATİK ROTA HESAPLAMASI ───────────────────────────────────
function getPose(s_val) {
    const s_mod = s_val % lapLen;
    let x = 0.0, y = 0.0, theta = 0.0, curv = 0.0;
    
    // 1. Düzlük: (0, 0) -> (400, 0)
    if (s_mod < straightLen) {
        x = s_mod;
        y = 0.0;
        theta = 0.0;
        curv = 0.0;
    }
    // 1. Viraj: (400, 0) -> (400, 100), merkez (400, 50)
    else if (s_mod < straightLen + semiCircleLen) {
        const s_rel = s_mod - straightLen;
        const phi = s_rel / radius; // 0 -> pi
        x = straightLen + radius * Math.sin(phi);
        y = radius - radius * Math.cos(phi);
        theta = phi;
        curv = 1.0 / radius;
    }
    // 2. Düzlük: (400, 100) -> (0, 100)
    else if (s_mod < 2 * straightLen + semiCircleLen) {
        const s_rel = s_mod - (straightLen + semiCircleLen);
        x = straightLen - s_rel;
        y = 2 * radius;
        theta = Math.PI;
        curv = 0.0;
    }
    // 2. Viraj: (0, 100) -> (0, 0), merkez (0, 50)
    else {
        const s_rel = s_mod - (2 * straightLen + semiCircleLen);
        const phi = s_rel / radius; // 0 -> pi
        x = -radius * Math.sin(phi);
        y = radius + radius * Math.cos(phi);
        theta = (Math.PI + phi) % (2 * Math.PI);
        curv = 1.0 / radius;
    }
    
    return { x, y, theta, curv };
}

// ─── 2. ÜÇ BOYUTLU NESNE YAPILANDIRMASI (THREE.JS) ───────────────────
function init() {
    const container = document.getElementById('canvas-container');
    
    // Sahne
    scene = new THREE.Scene();
    scene.background = new THREE.Color('#0a0c10');
    scene.fog = new THREE.FogExp2('#0a0c10', 0.005);
    
    // Kamera
    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.5, 1000);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Yer altına girmeyi engelle
    
    // Işıklandırma
    const ambientLight = new THREE.AmbientLight('#1e293b', 0.8);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight('#ffffff', 1.0);
    dirLight.position.set(100, 150, 50);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 500;
    const d = 200;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    scene.add(dirLight);

    // Grid / Zemin
    const grid = new THREE.GridHelper(1000, 100, '#1e293b', '#0f172a');
    grid.position.y = -0.05;
    scene.add(grid);
    
    const groundGeo = new THREE.PlaneGeometry(1200, 1200);
    const groundMat = new THREE.MeshStandardMaterial({ color: '#090d16', roughness: 0.9 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    
    // 3D Elemanları Üret
    build3DEnvironment();
    build3DBuses();
    
    // Event listener
    window.addEventListener('resize', onWindowResize);
    
    // Başlangıç Kamera Konumu
    camera.position.set(0, 100, 150);
    controls.update();
}

// ─── 3. 3D ŞEHİR / PİST YAPILANDIRMASI ─────────────────────────────────
function build3DEnvironment() {
    // 1. Rota/Asfalt Çizimi
    const pathSegments = 400;
    const roadPoints = [];
    const outerBorder = [];
    const innerBorder = [];
    
    for (let i = 0; i <= pathSegments; i++) {
        const s_p = (i / pathSegments) * lapLen;
        const pose = getPose(s_p);
        roadPoints.push(new THREE.Vector3(pose.x, 0.01, -pose.y)); // WebGL Z ekseni negatiftir
        
        // Şerit sınırları
        const nx = -Math.sin(pose.theta);
        const ny = Math.cos(pose.theta);
        outerBorder.push(new THREE.Vector3(pose.x + 3.6 * nx, 0.02, -(pose.y + 3.6 * ny)));
        innerBorder.push(new THREE.Vector3(pose.x - 3.6 * nx, 0.02, -(pose.y - 3.6 * ny)));
    }
    
    // Yol Çizgisini oluştur
    const roadGeom = new THREE.BufferGeometry().setFromPoints(roadPoints);
    const roadMat = new THREE.LineBasicMaterial({ color: '#ffcb2b', linewidth: 1 });
    const roadCenterLine = new THREE.LineSegments(roadGeom, new THREE.LineDashedMaterial({ 
        color: '#ffcb2b', dashSize: 4, gapSize: 4 
    }));
    roadCenterLine.computeLineDistances();
    scene.add(roadCenterLine);
    
    // Dış ve İç Sınırlar
    const outerGeom = new THREE.BufferGeometry().setFromPoints(outerBorder);
    const outerLine = new THREE.Line(outerGeom, new THREE.LineBasicMaterial({ color: '#cbd5e1', opacity: 0.5, transparent: true }));
    scene.add(outerLine);
    
    const innerGeom = new THREE.BufferGeometry().setFromPoints(innerBorder);
    const innerLine = new THREE.Line(innerGeom, new THREE.LineBasicMaterial({ color: '#cbd5e1', opacity: 0.5, transparent: true }));
    scene.add(innerLine);
    
    // 2. Asfalt Kaplanması (Kalın bant şeklinde)
    const asphaltGeom = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(roadPoints), 100, 3.6, 8, true);
    const asphaltMat = new THREE.MeshStandardMaterial({ color: '#1e293b', roughness: 0.95 });
    const asphalt = new THREE.Mesh(asphaltGeom, asphaltMat);
    asphalt.position.y = -0.1;
    scene.add(asphalt);
    
    // 3. Duraklar
    busStops.forEach((stop_s, idx) => {
        const pose = getPose(stop_s);
        const stopGeo = new THREE.CylinderGeometry(8.0, 8.0, 0.05, 32);
        const stopMat = new THREE.MeshStandardMaterial({ color: '#ef4444', transparent: true, opacity: 0.2 });
        const stopDisc = new THREE.Mesh(stopGeo, stopMat);
        stopDisc.position.set(pose.x, 0.02, -pose.y);
        scene.add(stopDisc);
        
        // Durak tabelası
        const poleGeo = new THREE.CylinderGeometry(0.1, 0.1, 5, 8);
        const poleMat = new THREE.MeshStandardMaterial({ color: '#64748b' });
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(pose.x - 4 * Math.sin(pose.theta), 2.5, -(pose.y + 4 * Math.cos(pose.theta)));
        scene.add(pole);
        
        const signGeo = new THREE.BoxGeometry(1.5, 1.0, 0.1);
        const signMat = new THREE.MeshStandardMaterial({ color: '#ef4444' });
        const sign = new THREE.Mesh(signGeo, signMat);
        sign.position.set(pole.position.x, 4.5, pole.position.z);
        sign.rotation.y = -pose.theta;
        scene.add(sign);
    });
    
    // 4. Landmarklar (LiDAR nirengi silindirleri)
    landmarksGroup = new THREE.Group();
    // Python'daki gibi 30m aralıklarla landmark koordinatı üret
    for (let s_l = 0.0; s_l < lapLen; s_l += 30.0) {
        const pose = getPose(s_l);
        const offset = 4.5;
        // Sol landmark
        let lx = pose.x - offset * Math.sin(pose.theta);
        let ly = pose.y + offset * Math.cos(pose.theta);
        createLandmarkCylinder(lx, -ly);
        
        // Sağ landmark
        lx = pose.x + offset * Math.sin(pose.theta);
        ly = pose.y - offset * Math.cos(pose.theta);
        createLandmarkCylinder(lx, -ly);
    }
    scene.add(landmarksGroup);
    
    // 5. Trafik Işığı (s = 400m -> x = 400.0, y = 0.0)
    const tlPoleGeo = new THREE.CylinderGeometry(0.15, 0.15, 6, 8);
    const tlPole = new THREE.Mesh(tlPoleGeo, new THREE.MeshStandardMaterial({ color: '#334155' }));
    tlPole.position.set(400.0, 3.0, 4.5);
    scene.add(tlPole);
    
    const tlBoxGeo = new THREE.BoxGeometry(0.8, 1.8, 0.8);
    const tlBoxMat = new THREE.MeshStandardMaterial({ color: '#0f172a' });
    const tlBox = new THREE.Mesh(tlBoxGeo, tlBoxMat);
    tlBox.position.set(400.0, 5.5, 4.5);
    scene.add(tlBox);
    
    const tlBulbGeo = new THREE.SphereGeometry(0.25, 16, 16);
    trafficLightMesh = new THREE.Mesh(tlBulbGeo, new THREE.MeshBasicMaterial({ color: '#ff1744' }));
    trafficLightMesh.position.set(399.55, 5.5, 4.5);
    scene.add(trafficLightMesh);
    
    // 6. Yaya Geçidi Zebra Çizgileri (s = 800m -> x = 157.08, y = 100.0)
    // simulator.js Z ekseni negatiftir, y = 100m -> z = -100m
    
    // 7. 3D Yaya (Pedestrian) Meshi
    const pedGroup = new THREE.Group();
    const pedHead = new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 16), new THREE.MeshStandardMaterial({ color: '#ffb74d' }));
    pedHead.position.y = 1.6;
    pedGroup.add(pedHead);
    
    const pedBody = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 1.0), new THREE.MeshStandardMaterial({ color: '#0288d1' }));
    pedBody.position.y = 0.9;
    pedGroup.add(pedBody);
    
    pedestrianMesh = pedGroup;
    pedestrianMesh.position.set(157.08, 0.0, -105.0); // Başlangıçta yolun dışında
    scene.add(pedestrianMesh);
}

function createLandmarkCylinder(x, z) {
    const geo = new THREE.CylinderGeometry(0.2, 0.2, 2.5, 8);
    const mat = new THREE.MeshStandardMaterial({ color: '#475569', roughness: 0.5 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, 1.25, z);
    mesh.castShadow = true;
    landmarksGroup.add(mesh);
}

// ─── 4. OTOBÜSLERİN (GERÇEK VE EKF GHOST) İLKLENDİRİLMESİ ────────────
function build3DBuses() {
    // AKIA Ultra LF12 EV otonom otobüsü (12.27m Boy, 2.54m En, 3.4m Yükseklik)
    // Fiziksel ölçeklendirme
    const length = 12.27;
    const width = 2.54;
    const height = 3.40;
    
    // 1. GERÇEK OTOBÜS (Ground Truth)
    bus = new THREE.Group();
    
    // Ana Şasi (Gövde) - Sleek Neon Mavi Kaplama
    const bodyGeo = new THREE.BoxGeometry(width, height - 0.6, length);
    const bodyMat = new THREE.MeshStandardMaterial({ 
        color: '#00e5ff', 
        roughness: 0.1, 
        metalness: 0.8,
        transparent: true,
        opacity: 0.85 
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = (height - 0.6) / 2 + 0.3; // Tekerlek boşluğu
    body.castShadow = true;
    body.receiveShadow = true;
    bus.add(body);
    
    // Ön Cam (Siyah Transparan)
    const glassGeo = new THREE.BoxGeometry(width - 0.02, 1.2, 0.5);
    const glassMat = new THREE.MeshStandardMaterial({ color: '#090d16', roughness: 0.0, metalness: 1.0, transparent: true, opacity: 0.9 });
    const windshield = new THREE.Mesh(glassGeo, glassMat);
    windshield.position.set(0, height - 1.0, length / 2 - 0.2);
    bus.add(windshield);
    
    // Tekerlekler (4 adet silindir)
    const wheelGeo = new THREE.CylinderGeometry(0.5, 0.5, 0.4, 24);
    const wheelMat = new THREE.MeshStandardMaterial({ color: '#0f172a', roughness: 0.9 });
    const wheelOffsets = [
        { x: -width/2, z: length/2 - 3.0 },  // Ön Sol (arka dingil referansı nedeniyle ön aks öndedir)
        { x: width/2,  z: length/2 - 3.0 },  // Ön Sağ
        { x: -width/2, z: -length/2 + 3.0 }, // Arka Sol
        { x: width/2,  z: -length/2 + 3.0 }  // Arka Sağ
    ];
    
    wheelOffsets.forEach(offset => {
        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(offset.x, 0.5, offset.z);
        wheel.castShadow = true;
        bus.add(wheel);
    });
    
    // Farlar (Headlights)
    const lightGeo = new THREE.SphereGeometry(0.15, 16, 16);
    const lightMat = new THREE.MeshBasicMaterial({ color: '#ffffff' });
    const headlightL = new THREE.Mesh(lightGeo, lightMat);
    headlightL.position.set(-width/2 + 0.3, 0.8, length/2);
    const headlightR = headlightL.clone();
    headlightR.position.x = width/2 - 0.3;
    bus.add(headlightL);
    bus.add(headlightR);
    
    // ── SENSÖR GÖRSEL ÖĞELERİNİN OTOBÜSE EKLENMESİ ────────────────────
    
    // A. LiDAR (Tavanın üstünde silindir)
    const lidarGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.3, 16);
    const lidarMat = new THREE.MeshStandardMaterial({ color: '#1e293b', metalness: 0.9 });
    const lidar = new THREE.Mesh(lidarGeo, lidarMat);
    lidar.position.set(0, height, length/2 - 4.0); // Tavan merkezi yakını
    bus.add(lidar);
    
    // LiDAR ışınları döngüsü
    const lidarLineMat = new THREE.LineBasicMaterial({ color: '#00e5ff', opacity: 0.35, transparent: true });
    for (let a = 0; a < 360; a += 15) {
        const angle = (a * Math.PI) / 180;
        const pts = [
            new THREE.Vector3(0, height, length/2 - 4.0),
            new THREE.Vector3(25 * Math.sin(angle), 0.05, length/2 - 4.0 + 25 * Math.cos(angle))
        ];
        const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const beam = new THREE.Line(lineGeom, lidarLineMat);
        bus.add(beam);
        lidarBeamLines.push(beam);
    }
    
    // LiDAR fütüristik yeşil/mavi tarama diski
    const lidarSweepGeo = new THREE.RingGeometry(0.1, 25.0, 32);
    const lidarSweepMat = new THREE.MeshBasicMaterial({ 
        color: '#00e5ff', 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.05 
    });
    const lidarSweep = new THREE.Mesh(lidarSweepGeo, lidarSweepMat);
    lidarSweep.rotation.x = -Math.PI / 2;
    lidarSweep.position.set(0, 0.1, length/2 - 4.0);
    bus.add(lidarSweep);
    
    // B. RADAR Konusu (Ön tamponda kırmızı transparan koni + tel kafes dış sınır)
    const radarCone = new THREE.ConeGeometry(5, 30, 16, 1, true);
    const radarConeMat = new THREE.MeshBasicMaterial({ 
        color: '#ff1744', 
        transparent: true, 
        opacity: 0.15, 
        side: THREE.DoubleSide
    });
    radarConeMesh = new THREE.Mesh(radarCone, radarConeMat);
    radarConeMesh.rotation.x = Math.PI / 2; // Ön yöne yönelt
    radarConeMesh.position.set(0, 0.6, length/2 + 15);
    
    // Tel kafes RADAR kaplaması
    const radarWireMat = new THREE.MeshBasicMaterial({ color: '#ff1744', wireframe: true, transparent: true, opacity: 0.35 });
    const radarWire = new THREE.Mesh(radarCone, radarWireMat);
    radarConeMesh.add(radarWire);
    
    bus.add(radarConeMesh);
    
    // C. Çevre Kameraları (6 FOV Frustumu)
    // 6 Yön: Front, Rear, Left, Right, Front-Left Blind, Front-Right Blind
    const camAngles = [
        { name: 'front', rot: 0, scale: [2, 2, 8], pos: [0, 2.5, length/2], color: '#00e5ff' },
        { name: 'rear', rot: Math.PI, scale: [2, 2, 8], pos: [0, 2.5, -length/2], color: '#7b1fa2' },
        { name: 'left', rot: Math.PI/2, scale: [2, 2, 7], pos: [-width/2, 2.5, 0], color: '#f57f17' },
        { name: 'right', rot: -Math.PI/2, scale: [2, 2, 7], pos: [width/2, 2.5, 0], color: '#2e7d32' },
        { name: 'front_left_blind', rot: Math.PI/4, scale: [1.8, 1.8, 4], pos: [-width/2, 1.5, length/2 - 1], color: '#c62828' },
        { name: 'front_right_blind', rot: -Math.PI/4, scale: [1.8, 1.8, 4], pos: [width/2, 1.5, length/2 - 1], color: '#ad1457' }
    ];
    
    camAngles.forEach(cam => {
        const frustumGeo = new THREE.ConeGeometry(cam.scale[0], cam.scale[2], 4);
        const frustumMat = new THREE.MeshBasicMaterial({ color: cam.color, transparent: true, opacity: 0.10 });
        const frustum = new THREE.Mesh(frustumGeo, frustumMat);
        frustum.rotation.x = Math.PI / 2;
        frustum.rotation.z = cam.rot;
        frustum.position.set(cam.pos[0], cam.pos[1], cam.pos[2]);
        // Koninin sivri ucu araçta olsun, tabanı dışa baksın
        frustum.translateY(cam.scale[2]/2);
        
        // Kameralar için tel kafes frustum dış çizgileri
        const frustumWireMat = new THREE.MeshBasicMaterial({ color: cam.color, wireframe: true, transparent: true, opacity: 0.25 });
        const frustumWire = new THREE.Mesh(frustumGeo, frustumWireMat);
        frustum.add(frustumWire);
        
        bus.add(frustum);
        cameraFrustums.push(frustum);
    });
    
    // D. Bumper Ultrasonik Sensörler (12 Adet mini yay halkası)
    const ultraOffsets = [
        { x: -width/2, z: length/2, angle: Math.PI/4 },
        { x: 0, z: length/2, angle: 0 },
        { x: width/2, z: length/2, angle: -Math.PI/4 },
        { x: -width/2, z: -length/2, angle: 3*Math.PI/4 },
        { x: 0, z: -length/2, angle: Math.PI },
        { x: width/2, z: -length/2, angle: -3*Math.PI/4 }
    ];
    
    ultraOffsets.forEach(uo => {
        // Küçük çember kesiti ile ultrasonik yayılımı simüle et (parlak yeşil)
        const arcGeo = new THREE.RingGeometry(0.1, 2.5, 8, 1, uo.angle - 0.2, 0.4);
        const arcMat = new THREE.MeshBasicMaterial({ color: '#00e676', side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
        const arc = new THREE.Mesh(arcGeo, arcMat);
        arc.rotation.x = -Math.PI / 2;
        arc.position.set(uo.x, 0.2, uo.z);
        bus.add(arc);
        ultrasonicArcs.push(arc);
    });
    
    // E. RTK GNSS Anteni (İnce dikey mavi lazer silindiri - gökyüzüne uzanır)
    const gnssLineGeo = new THREE.CylinderGeometry(0.04, 0.04, 40.0, 8);
    const gnssLineMat = new THREE.MeshBasicMaterial({ color: '#2196f3', transparent: true, opacity: 0.4 });
    gnssSatelliteLine = new THREE.Mesh(gnssLineGeo, gnssLineMat);
    gnssSatelliteLine.position.set(0, height + 20.0, length/2 - 1.0);
    bus.add(gnssSatelliteLine);
    
    scene.add(bus);
    
    // 2. EKF GHOST OTOBÜS (Transparan Mavi - EKF Tahmini)
    ghostBus = new THREE.Group();
    const gBodyGeo = new THREE.BoxGeometry(width, height - 0.6, length);
    const gBodyMat = new THREE.MeshStandardMaterial({ 
        color: '#1565c0', 
        roughness: 0.5,
        transparent: true,
        opacity: 0.35,
        wireframe: true
    });
    const gBody = new THREE.Mesh(gBodyGeo, gBodyMat);
    gBody.position.y = (height - 0.6) / 2 + 0.3;
    ghostBus.add(gBody);
    scene.add(ghostBus);
    
    // 3. EKF 2-Sigma Konum Belirsizlik Çemberi (Otobüs altında mavi disk)
    const errDiscGeo = new THREE.RingGeometry(0.1, 2.5, 32);
    const errDiscMat = new THREE.MeshBasicMaterial({ color: '#2196f3', side: THREE.DoubleSide, transparent: true, opacity: 0.15 });
    EKF_UncertaintyDisk = new THREE.Mesh(errDiscGeo, errDiscMat);
    EKF_UncertaintyDisk.rotation.x = -Math.PI / 2;
    EKF_UncertaintyDisk.position.y = 0.05;
    scene.add(EKF_UncertaintyDisk);
    
    // Yol geçmiş çizgileri
    trajectoryLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#ffffff', opacity: 0.3 }));
    scene.add(trajectoryLine);
    
    ekfLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#2196f3', opacity: 0.6 }));
    scene.add(ekfLine);
}

// ─── 5. PENCERE BOYUTLANDIRMASI ───────────────────────────────────────
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// ─── 6. HUD EKRANI VE KONTROLLERİNİN GÜNCELLEMESİ ────────────────────
function setCameraMode(mode) {
    cameraMode = mode;
    // Buton aktiflik sınıflarını temizle ve ata
    ['chase', 'top', 'orbit', 'cabin'].forEach(m => {
        const btn = document.getElementById(`btn-cam-${m}`);
        if (btn) btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`btn-cam-${mode}`);
    if (activeBtn) activeBtn.classList.add('active');
}

function toggleSensor(sensor) {
    activeSensors[sensor] = !activeSensors[sensor];
    const row = document.getElementById(`sensor-${sensor}`);
    if (row) {
        if (activeSensors[sensor]) {
            row.classList.add('active');
        } else {
            row.classList.remove('active');
        }
    }
    
    // 3D Görsel kapatmaları
    if (sensor === 'lidar') {
        lidarBeamLines.forEach(beam => beam.visible = activeSensors.lidar);
    } else if (sensor === 'radar') {
        radarConeMesh.visible = activeSensors.radar;
    } else if (sensor === 'camera') {
        cameraFrustums.forEach(fr => fr.visible = activeSensors.camera);
    } else if (sensor === 'ultrasonic') {
        ultrasonicArcs.forEach(arc => arc.visible = activeSensors.ultrasonic);
    } else if (sensor === 'rtk') {
        gnssSatelliteLine.visible = activeSensors.rtk;
    }
}

function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('btn-pause');
    if (isPaused) {
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Devam Et';
    } else {
        btn.innerHTML = '<i class="fa-solid fa-pause"></i> Duraklat';
    }
}

function resetSim() {
    s = 0.0;
    v = 0.0;
    t = 0.0;
    truePathPoints = [];
    ekfPathPoints = [];
    ekfX = 0; ekfY = 0; ekfTheta = 0;
    isPaused = false;
    const btn = document.getElementById('btn-pause');
    if (btn) btn.innerHTML = '<i class="fa-solid fa-pause"></i> Duraklat';
}

// ─── 7. FİZİK VE CANLI SİMÜLASYON DÖNGÜSÜ ─────────────────────────────
const dt = 0.033; // ~30 FPS

function animate() {
    requestAnimationFrame(animate);
    
    if (!isPaused) {
        t += dt;
        
        // ─── A. Trafik Işığı ve Yaya Zaman Döngüsü ────────────────────
        const cycleTime = t % 30.0;
        if (cycleTime < 12.0) {
            tlColor = "green";
            trafficLightMesh.material.color.setHex(0x00e676); // Yeşil
        } else if (cycleTime < 15.0) {
            tlColor = "yellow";
            trafficLightMesh.material.color.setHex(0xffb74d); // Sarı
        } else {
            tlColor = "red";
            trafficLightMesh.material.color.setHex(0xff1744); // Kırmızı
        }
        
        // Yaya döngüsü (65 - 75 saniyeler arası aktif)
        pedActive = (t >= 35.0 && t <= 45.0) || (t >= 95.0 && t <= 105.0); // 3D'de daha hızlı görmek için süreleri sıklaştırdık
        
        // Yayanın 3B hareketini güncelle
        if (pedActive) {
            let relativeT = (t % 60.0);
            let pedZ = -105.0; // Varsayılan yol dışı
            
            // Periyoduna göre yaya yola girsin
            if (relativeT >= 35.0 && relativeT <= 45.0) {
                const subT = relativeT - 35.0;
                if (subT <= 2.0) {
                    pedZ = -105.0 + 5.0 * (subT / 2.0); // Yola giriyor
                } else if (subT <= 8.0) {
                    pedZ = -100.0; // Yolu bloke ediyor
                } else {
                    pedZ = -100.0 - 5.0 * ((subT - 8.0) / 2.0); // Yoldan çıkıyor
                }
            } else if (relativeT >= 95.0 && relativeT <= 105.0) {
                const subT = relativeT - 95.0;
                if (subT <= 2.0) {
                    pedZ = -105.0 + 5.0 * (subT / 2.0);
                } else if (subT <= 8.0) {
                    pedZ = -100.0;
                } else {
                    pedZ = -100.0 - 5.0 * ((subT - 8.0) / 2.0);
                }
            }
            pedestrianMesh.position.z = pedZ;
        } else {
            pedestrianMesh.position.z = -105.0; // Yol dışı
        }
        
        // ─── B. Hız ve Kinematik Güncelleme ───────────────────────────
        const pose = getPose(s);
        let v_nominal = pose.curv > 0 ? 4.0 : 8.0; // Virajlarda 14 km/h, düzlükte 28 km/h
        
        // Sensör Kararları (Işık ve Yaya tespiti)
        let d_light = 400.0 - (s % lapLen);
        if (d_light < -lapLen/2) d_light += lapLen;
        else if (d_light > lapLen/2) d_light -= lapLen;
        
        let tl_stop = false;
        if (tlColor !== "green" && d_light > 0 && d_light <= 30.0) {
            tl_stop = true;
        }
        
        let d_ped = 800.0 - (s % lapLen);
        if (d_ped < -lapLen/2) d_ped += lapLen;
        else if (d_ped > lapLen/2) d_ped -= lapLen;
        
        let ped_stop = false;
        if (pedActive && d_ped > 0 && d_ped <= 25.0) {
            ped_stop = true;
        }
        
        // Hedef Hız Seçimi
        let v_target = v_nominal;
        let isEmergency = false;
        if (ped_stop) {
            v_target = Math.max(0.0, ((d_ped - 3.0) / 22.0) * v_nominal);
            isEmergency = true;
        } else if (tl_stop) {
            v_target = Math.max(0.0, ((d_light - 2.5) / 27.5) * v_nominal);
        }
        
        // Durma kilidi
        if ((tl_stop && d_light <= 3.0) || (ped_stop && d_ped <= 3.5)) {
            v_target = 0.0;
            v = 0.0;
        } else {
            // İvmelenme sınırları
            const acc = v_target > v ? 1.2 : -1.8;
            v = Math.max(0.0, Math.min(8.0, v + acc * dt));
            s += v * dt;
        }
        
        // ─── C. 3D Otobüs Pozisyon/Yönelim Güncelleme ─────────────────
        const nextPose = getPose(s);
        bus.position.set(nextPose.x, 0.0, -nextPose.y);
        bus.rotation.y = nextPose.theta + Math.PI / 2; // WebGL yaw yönü + 90 derece ofset düzeltmesi
        
        // Dönen Tekerlek Animasyonu
        bus.children.forEach(child => {
            if (child.geometry && child.geometry.type === 'CylinderGeometry' && child.position.y === 0.5) {
                // Ön tekerlekleri direksiyona göre hafif döndür
                if (child.position.z > 0) {
                    child.rotation.y = (pose.curv > 0) ? Math.PI/6 : 0;
                }
                child.rotation.x += v * dt * 2.0; // Hıza bağlı dönüş hızı
            }
        });
        
        // ─── D. Gerçek Zamanlı EKF / Kestirim Güncelleme ─────────────
        // Gerçek konuma gürültü ekleyerek EKF kestirimi simüle et
        const noiseX = (Math.random() - 0.5) * 0.08;
        const noiseY = (Math.random() - 0.5) * 0.08;
        
        // AEKF varyans adaptasyonu: tünelde/bozulmada hata büyür
        let is_degraded = (s % lapLen >= 300 && s % lapLen <= 500);
        let errorScale = is_degraded ? 4.0 : 1.0;
        ekfError = 0.03 * errorScale + Math.sin(t*0.5)*0.01;
        
        ekfX = nextPose.x + noiseX * errorScale;
        ekfY = nextPose.y + noiseY * errorScale;
        ekfTheta = nextPose.theta + (Math.random() - 0.5) * 0.01 * errorScale;
        
        // Ghost Bus pozisyonu
        ghostBus.position.set(ekfX, 0.0, -ekfY);
        ghostBus.rotation.y = ekfTheta + Math.PI / 2;
        
        // EKF Belirsizlik Diski boyutu
        const scaleRadius = 1.0 + ekfError * 15.0;
        EKF_UncertaintyDisk.position.set(ekfX, 0.05, -ekfY);
        EKF_UncertaintyDisk.scale.set(scaleRadius, scaleRadius, 1.0);
        
        // LiDAR animasyonu döndürme
        if (activeSensors.lidar) {
            lidarBeamLines.forEach(beam => {
                beam.rotation.y += 0.05;
            });
        }
        
        // Radar animasyonu daralma/büyüme
        if (activeSensors.radar) {
            if (isEmergency && d_ped < 15) {
                radarConeMesh.scale.set(0.6, 0.6, 0.4);
                radarConeMesh.children[0].material.color.setHex(0xff1744); // Tel kafes kırmızı
                radarConeMesh.material.color.setHex(0xff1744); // Parlak Kırmızı
            } else {
                radarConeMesh.scale.set(1.0, 1.0, 1.0);
                radarConeMesh.children[0].material.color.setHex(0x00e5ff); // Tel kafes mavi
                radarConeMesh.material.color.setHex(0x00e5ff); // Normal Mavi/Cyan ACC tarayıcı
            }
        }
        
        // Ultrasonik sonar sensör dalgalanması (pulsing waves)
        if (activeSensors.ultrasonic) {
            ultrasonicArcs.forEach((arc, u_idx) => {
                const waveScale = 1.0 + ((t * 8 + u_idx) % 3) * 0.4;
                arc.scale.set(waveScale, waveScale, 1.0);
                arc.material.opacity = 0.4 - (waveScale - 1.0) * 0.3;
            });
        }
        
        // Yol Geçmişini kaydet
        if (Math.floor(t * 10) % 2 === 0) {
            truePathPoints.push(new THREE.Vector3(nextPose.x, 0.05, -nextPose.y));
            ekfPathPoints.push(new THREE.Vector3(ekfX, 0.06, -ekfY));
            if (truePathPoints.length > 250) {
                truePathPoints.shift();
                ekfPathPoints.shift();
            }
            
            trajectoryLine.geometry.setFromPoints(truePathPoints);
            ekfLine.geometry.setFromPoints(ekfPathPoints);
        }
        
        // ─── E. HUD Telemetri Paneli Metinleri Güncelleme ─────────────
        document.getElementById('val-speed').innerText = (v * 3.6).toFixed(1);
        document.getElementById('val-distance').innerText = s.toFixed(1);
        document.getElementById('val-steer').innerText = (pose.curv * radius * 180 / Math.PI).toFixed(1);
        document.getElementById('val-accel').innerText = (v_target > v ? 1.2 : v_target < v ? -1.8 : 0.0).toFixed(1);
        
        // Trafik ışık durumu
        const tlBadge = document.getElementById('status-light');
        tlBadge.innerText = tlColor.toUpperCase();
        tlBadge.className = 'badge ' + (tlColor === 'green' ? 'badge-green' : tlColor === 'yellow' ? 'badge-yellow' : 'badge-red');
        
        // Yaya durumu
        const pedBadge = document.getElementById('status-ped');
        if (pedActive) {
            const blocked = (t % 60.0 >= 37.0 && t % 60.0 <= 43.0) || (t % 60.0 >= 97.0 && t % 60.0 <= 103.0);
            pedBadge.innerText = blocked ? 'YOL BLOKE!' : 'YAKLAŞIYOR';
            pedBadge.className = 'badge ' + (blocked ? 'badge-red' : 'badge-yellow');
        } else {
            pedBadge.innerText = 'SERBEST';
            pedBadge.className = 'badge badge-green';
        }
        
        // Dwell / Durak durumu
        const dwellBadge = document.getElementById('status-dwell');
        let currentDwellStatus = "CRUISE";
        busStops.forEach(stop_s => {
            if (Math.abs(s % lapLen - stop_s) < 2.5) {
                currentDwellStatus = "DURAKTA (DWELL)";
            }
        });
        dwellBadge.innerText = currentDwellStatus;
        dwellBadge.className = 'badge ' + (currentDwellStatus === 'CRUISE' ? '' : 'badge-yellow');
        
        // EKF hata metni
        document.getElementById('val-ekf-err').innerText = `${ekfError.toFixed(3)} m`;
        document.getElementById('val-gnss-err').innerText = is_degraded ? 'Zayıf (Kanyon)' : 'Normal (RTK)';
        document.getElementById('val-gnss-err').style.color = is_degraded ? '#ff1744' : '#00e676';
        
        // Radar HUD metni
        const radarHUD = document.getElementById('radar-val');
        if (ped_stop) {
            radarHUD.innerText = `YAYA TESPİT EDİLDİ! Mesafe: ${d_ped.toFixed(1)}m (AEB FREN)`;
            radarHUD.style.color = '#ff1744';
        } else if (tl_stop) {
            radarHUD.innerText = `TRAFİK IŞIĞI TESPİTİ! Mesafe: ${d_light.toFixed(1)}m (YAVAŞLAMA)`;
            radarHUD.style.color = '#ff9100';
        } else {
            radarHUD.innerText = "Mesafe: Güvenli (>50m)";
            radarHUD.style.color = '#cbd5e1';
        }
    }
    
    // ─── H. Kamera Takip Modu Güncellemeleri ────────────────────────
    const busPos = bus.position;
    const busRot = bus.rotation.y;
    
    if (cameraMode === 'chase') {
        const offset = new THREE.Vector3(0, 10, -22); // Otobüsün arkasında ve üstünde
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), busRot);
        camera.position.copy(busPos).add(offset);
        camera.lookAt(busPos.clone().add(new THREE.Vector3(0, 2.5, 0)));
        controls.target.copy(busPos);
    } else if (cameraMode === 'top') {
        camera.position.set(busPos.x, 90, busPos.z);
        camera.lookAt(busPos);
        controls.target.copy(busPos);
    } else if (cameraMode === 'cabin') {
        // Ön cam / Sürücü kabin görüşü
        const cabinPos = new THREE.Vector3(0, 2.4, 4.5); // Sürücü koltuğu konumu
        cabinPos.applyAxisAngle(new THREE.Vector3(0, 1, 0), busRot);
        camera.position.copy(busPos).add(cabinPos);
        
        const targetOffset = new THREE.Vector3(0, 0, 30);
        targetOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), busRot);
        camera.lookAt(busPos.clone().add(targetOffset));
        controls.target.copy(busPos.clone().add(targetOffset));
    }
    
    controls.update();
    renderer.render(scene, camera);
}

// Başlat
init();
animate();
