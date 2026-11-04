// Camera của cameras.json (src/product/export_web.py) -> camera three.js, cùng hệ tọa độ với model.splat.
// Quy ước OpenCV của file: mỗi khung có position, forward (= trục z của camera), up (= -y của camera).
// three.js nhìn theo -Z cục bộ, +Y lên, +X sang phải: X = right = forward × up, Y = up, Z = -forward.
import * as THREE from "three";

export function vec(values) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

// Hướng (quaternion) của camera three.js khi đặt đúng như một khung ảnh
export function frameQuaternion(forward, up) {
  const f = forward.clone().normalize();
  const u = up.clone().sub(f.clone().multiplyScalar(up.dot(f))).normalize();  // trực giao hóa
  const r = new THREE.Vector3().crossVectors(f, u).normalize();
  const m = new THREE.Matrix4().makeBasis(r, u, f.clone().negate());
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

export function frameOf(cameras, index) {
  const frame = cameras.frames[index];
  return { name: frame.name, position: vec(frame.position), forward: vec(frame.forward).normalize(), up: vec(frame.up).normalize() };
}

// Đặt camera three.js đúng vị trí, hướng và góc nhìn dọc của khung ảnh
export function placeAtFrame(camera, cameras, index) {
  const frame = frameOf(cameras, index);
  camera.position.copy(frame.position);
  camera.quaternion.copy(frameQuaternion(frame.forward, frame.up));
  camera.up.copy(vec(cameras.up)).normalize();
  if (cameras.intrinsics?.fov_y_deg) camera.fov = cameras.intrinsics.fov_y_deg;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return frame;
}

// Điểm 3D -> pixel của ảnh theo đúng intrinsics trong cameras.json (dùng để kiểm tra camera khớp mô hình)
export function projectToImage(cameras, index, point) {
  const frame = frameOf(cameras, index);
  const k = cameras.intrinsics;
  const right = new THREE.Vector3().crossVectors(frame.forward, frame.up).normalize();
  const d = vec(point).sub(frame.position);
  const z = d.dot(frame.forward);
  if (z <= 0) return null;
  return [k.fx * d.dot(right) / z + k.cx, k.fy * -d.dot(frame.up) / z + k.cy];
}

// Hệ trục để đi bộ / vẽ bản đồ nhỏ: up của phòng, và hai trục nằm ngang (e1 theo hướng trải rộng nhất của đường quay)
export function groundBasis(cameras) {
  const up = vec(cameras.up).normalize();
  const positions = cameras.frames.map((f) => vec(f.position));
  const mean = positions.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / Math.max(1, positions.length));
  let e1 = new THREE.Vector3(1, 0, 0);
  if (Math.abs(e1.dot(up)) > 0.9) e1.set(0, 0, 1);
  e1.sub(up.clone().multiplyScalar(e1.dot(up))).normalize();
  // vài vòng lặp lũy thừa trên hiệp phương sai của vị trí (đã chiếu xuống mặt ngang)
  const flat = positions.map((p) => {
    const d = p.clone().sub(mean);
    return d.sub(up.clone().multiplyScalar(d.dot(up)));
  });
  if (flat.length > 2) {
    for (let iter = 0; iter < 8; iter++) {
      const next = new THREE.Vector3();
      for (const d of flat) next.add(d.clone().multiplyScalar(d.dot(e1)));
      next.sub(up.clone().multiplyScalar(next.dot(up)));
      if (next.lengthSq() < 1e-12) break;
      e1 = next.normalize();
    }
  }
  const e2 = new THREE.Vector3().crossVectors(up, e1).normalize();
  const heights = positions.map((p) => p.dot(up));
  return { up, e1, e2, mean, minHeight: Math.min(...heights), maxHeight: Math.max(...heights) };
}

// Vùng đã quay: hộp bao các vị trí camera (nới thêm margin), giao với hộp bao Gaussian (bounds)
export function walkBox(cameras, marginRatio = 0.25) {
  const box = new THREE.Box3();
  for (const frame of cameras.frames) box.expandByPoint(vec(frame.position));
  box.expandByScalar((cameras.scene_radius || 1) * marginRatio);
  if (cameras.bounds?.min && cameras.bounds?.max) {
    const bounds = new THREE.Box3(vec(cameras.bounds.min), vec(cameras.bounds.max));
    const clipped = box.clone().intersect(bounds);
    if (!clipped.isEmpty()) return clipped;
  }
  return box;
}

// Hướng nhìn làm mượt (trung bình trượt) cho chế độ tham quan
export function smoothedFrames(cameras, window = 5) {
  const frames = cameras.frames.map((_, i) => frameOf(cameras, i));
  const half = Math.max(0, Math.floor(window / 2));
  return frames.map((frame, i) => {
    const forward = new THREE.Vector3();
    const up = new THREE.Vector3();
    for (let j = Math.max(0, i - half); j <= Math.min(frames.length - 1, i + half); j++) {
      forward.add(frames[j].forward);
      up.add(frames[j].up);
    }
    return { ...frame, forward: forward.normalize(), up: up.normalize() };
  });
}
