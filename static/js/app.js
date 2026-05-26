const API = '';   // same origin

/* ═══════════════════════════════════════════
   STATE
═══════════════════════════════════════════ */
let allEmployees = [];         // full list from server
let editCode     = null;       // employee_code being edited
let editSigs     = [];         // existing signatures for edit page

// file state: { c: {1: File|null, 2: File|null}, e: {1:…, 2:…} }
const files = { c: { 1: null, 2: null }, e: { 1: null, 2: null } };

/* ═══════════════════════════════════════════
   ROUTING
═══════════════════════════════════════════ */
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
}

function goList() {
  setBreadcrumb([]);
  showPage('list');
  loadList();
}

function goCreate() {
  setBreadcrumb([
    { label: 'Danh sách', action: 'goList()' },
    { label: 'Thêm nhân viên' },
  ]);
  resetCreateForm();
  showPage('create');
}

function goEdit(code) {
  editCode = code;
  setBreadcrumb([
    { label: 'Danh sách', action: 'goList()' },
    { label: `Chỉnh sửa: ${code}` },
  ]);
  loadEditForm(code);
  showPage('edit');
}

/* ═══════════════════════════════════════════
   BREADCRUMB
═══════════════════════════════════════════ */
function setBreadcrumb(items) {
  const bc = document.getElementById('breadcrumb');
  if (!items.length) { bc.innerHTML = ''; return; }
  bc.innerHTML = items.map((it, i) => {
    const isLast = i === items.length - 1;
    const html = isLast
      ? `<span>${it.label}</span>`
      : `<span class="crumb-link" onclick="${it.action}">${it.label}</span>`;
    return i === 0 ? html : `<span class="sep">›</span>${html}`;
  }).join('');
}

/* ═══════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════ */
function setStep(prefix, n, state) {
  const el = document.getElementById(prefix + n);
  if (!el) return;
  el.classList.remove('active','done');
  if (state === 'active') el.classList.add('active');
  if (state === 'done')   el.classList.add('done');
}

function setFieldErr(inputId, errId, show) {
  document.getElementById(inputId).classList.toggle('err', show);
  document.getElementById(errId).style.display = show ? 'block' : 'none';
}

function setLoading(ctx, loading) {
  const prefix = ctx === 'c' ? 'c' : 'e';
  document.getElementById(prefix+'SubmitBtn').disabled = loading;
  document.getElementById(prefix+'Spinner').style.display  = loading ? 'block' : 'none';
  document.getElementById(prefix+'SubmitLabel').textContent = loading
    ? 'Đang xử lý…'
    : (ctx === 'c' ? 'Tạo nhân viên' : 'Lưu thay đổi');
}

function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

let _toastTimer;
function showToast(type, title, msg) {
  const toast = document.getElementById('toast');
  document.getElementById('tIcon').textContent  = type === 'success' ? '✓' : '✕';
  document.getElementById('tTitle').textContent = title;
  document.getElementById('tMsg').textContent   = msg;
  toast.className = type;
  toast.style.display = 'flex';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 5000);
}

/* ═══════════════════════════════════════════
   SIAMESE PANEL
═══════════════════════════════════════════ */
async function loadSiameseStatus() {
  try {
    const res  = await fetch(`${API}/siamese/status`);
    const data = await res.json();
    const el   = document.getElementById('siameseStatus');
    if (data.model_available) {
      const dt = data.last_trained
        ? new Date(data.last_trained).toLocaleString('vi-VN')
        : 'không rõ';
      el.textContent =
        `✓ Model đã huấn luyện — lần cuối: ${dt} ` +
        `(${data.n_employees_at_training ?? '?'} NV, ${data.n_pairs_at_training ?? '?'} cặp)`;
      el.style.color = 'var(--green)';
    } else {
      el.textContent = '⚠ Chưa có model — nhấn Huấn luyện để bắt đầu';
      el.style.color = 'var(--gray-500)';
    }
  } catch {
    document.getElementById('siameseStatus').textContent = 'Không thể lấy trạng thái';
  }
}

async function trainSiamese() {
  const btn    = document.getElementById('siameseTrainBtn');
  const spin   = document.getElementById('siameseSpinner');
  const label  = document.getElementById('siameseTrainLabel');

  btn.disabled = true;
  spin.style.display = 'block';
  label.textContent  = 'Đang huấn luyện…';
  document.getElementById('siameseStatus').textContent = 'Đang huấn luyện, vui lòng chờ…';
  document.getElementById('siameseStatus').style.color = 'var(--gray-500)';

  try {
    const res  = await fetch(`${API}/siamese/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 60, augment_factor: 15 }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);

    if (data.status === 'ok') {
      showToast('success', 'Huấn luyện hoàn tất!',
        `${data.n_pairs} cặp, ${data.epochs_run} epochs, ` +
        `val loss=${data.best_val_loss.toFixed(4)}, ` +
        `thời gian=${data.duration_seconds}s`);
    } else {
      showToast('error', 'Không thể huấn luyện', data.message);
    }
  } catch (err) {
    showToast('error', 'Lỗi huấn luyện', err.message);
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
    label.textContent  = 'Huấn luyện lại';
    loadSiameseStatus();
  }
}

/* ═══════════════════════════════════════════
   INIT
═══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  loadList();
  loadSiameseStatus();
});
