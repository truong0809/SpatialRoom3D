// Gọi API của server (app/web/server.py). Mọi request ghi kèm header X-SR3D: 1 (server từ chối nếu thiếu).

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data || {};
  }
}

let unauthorizedHandler = () => {};

export function onUnauthorized(handler) {
  unauthorizedHandler = handler;
}

export async function api(path, { method = "GET", json, body, signal, headers = {} } = {}) {
  const init = { method, headers: { ...headers }, credentials: "same-origin", signal, cache: "no-store" };
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    init.headers["Content-Type"] = "application/json";
  } else if (body !== undefined) {
    init.body = body;
  }
  if (method !== "GET" && method !== "HEAD") init.headers["X-SR3D"] = "1";
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("Không kết nối được tới server (pod tắt, mạng ngắt, hoặc server chưa chạy)", 0);
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (response.status === 401 && path !== "/api/login") unauthorizedHandler();
  if (!response.ok) {
    throw new ApiError((data && data.error) || `Lỗi ${response.status}`, response.status, data);
  }
  return data;
}

export const status = () => api("/api/status");
export const rooms = () => api("/api/rooms");
export const room = (id) => api(`/api/rooms/${encodeURIComponent(id)}`);
export const roomLog = (id, lines = 200) => api(`/api/rooms/${encodeURIComponent(id)}/log?lines=${lines}`);
export const cancelRoom = (id) => api(`/api/rooms/${encodeURIComponent(id)}/cancel`, { method: "POST", json: {} });
export const login = (token) => api("/api/login", { method: "POST", json: { token } });
