/* ═══════════════════════════════════════════
   C4 CUSTOMERS — STATE
═══════════════════════════════════════════ */
let allC4Records   = [];
let c4PdfFile      = null;
let c4Extracted    = null;  // result from /extract-preview
let c4DetailId     = null;

/* ── Navigation ── */
function goC4List() {
  setBreadcrumb([{ label: 'Danh sách KH' }]);
  showPage('c4-list');
  loadC4List();
}

function goC4Create() {
  setBreadcrumb([
    { label: 'Danh sách KH', action: 'goC4List()' },
    { label: 'Thêm khách hàng' },
  ]);
  resetC4CreateForm();
  showPage('c4-create');
}

function goC4Detail(id) {
  c4DetailId = id;
  setBreadcrumb([
    { label: 'Danh sách KH', action: 'goC4List()' },
    { label: `Chi tiết #${id}` },
  ]);
  loadC4Detail(id);
  showPage('c4-detail');
}

/* ── List ── */
async function loadC4List() {
  renderC4Skeleton();
  try {
    const res = await fetch(`${API}/c4-customers?limit=200`);
    allC4Records = await res.json();
    renderC4Table(allC4Records);
  } catch {
    showToast('error', 'Lỗi tải dữ liệu', 'Không thể kết nối tới server.');
    renderC4Table([]);
  }
}

function renderC4Skeleton() {
  const tbody = document.getElementById('c4Tbody');
  tbody.innerHTML = Array(4).fill(0).map(() => `
    <tr class="skel-row">
      <td><div class="skeleton" style="width:20px"></div></td>
      <td><div class="skeleton" style="width:80px"></div></td>
      <td><div class="skeleton" style="width:150px"></div></td>
      <td><div class="skeleton" style="width:120px"></div></td>
      <td><div class="skeleton" style="width:90px"></div></td>
      <td><div class="skeleton" style="width:60px"></div></td>
      <td><div class="skeleton" style="width:80px"></div></td>
      <td><div class="skeleton" style="width:100px"></div></td>
    </tr>`).join('');
}

function renderC4Table(rows) {
  const tbody = document.getElementById('c4Tbody');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8">
      <div class="empty-state">
        <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/></svg>
        <p>Chưa có khách hàng nào. <span class="crumb-link" onclick="goC4Create()">Thêm ngay</span></p>
      </div>
    </td></tr>`;
    document.getElementById('c4TableMeta').textContent = '';
    return;
  }

  tbody.innerHTML = rows.map((r, i) => {
    const sigCount = Object.values(r.signatures).filter(Boolean).length;
    const date = new Date(r.created_at).toLocaleDateString('vi-VN');
    const badge = sigCount > 0
      ? `<span class="badge badge-sig">✓ ${sigCount}</span>`
      : `<span class="badge badge-nosig">0</span>`;
    return `<tr style="cursor:pointer" onclick="goC4Detail(${r.id})">
      <td style="color:var(--gray-400);font-size:.8rem">${i + 1}</td>
      <td style="font-size:.82rem;font-weight:600;color:var(--primary)">${esc(r.ma_khach_hang || '—')}</td>
      <td><strong>${esc(r.ten_dang_ky_kinh_doanh || '—')}</strong></td>
      <td>${esc(r.ten_cua_hang || '—')}</td>
      <td style="font-size:.82rem;color:var(--gray-600)">${esc(r.giay_phep_so || '—')}</td>
      <td>${badge}</td>
      <td style="color:var(--gray-500);font-size:.82rem">${date}</td>
      <td>
        <div class="actions-cell" onclick="event.stopPropagation()">
          <button class="btn-icon check" onclick="openDoValidateModal(${r.id},'${esc(r.ten_dang_ky_kinh_doanh || String(r.id))}')" ${sigCount > 0 ? '' : 'disabled title="Cần có chữ ký mẫu"'}>
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            Validate DO
          </button>
          <button class="btn-icon edit" onclick="goC4Detail(${r.id})">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
            Xem
          </button>
          <button class="btn-icon delete" onclick="confirmDeleteC4Record(${r.id},'${esc(r.ten_dang_ky_kinh_doanh || String(r.id))}')">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/></svg>
            Xóa
          </button>
        </div>
      </td>
    </tr>`;
  }).join('');

  document.getElementById('c4TableMeta').textContent =
    `Hiển thị ${rows.length} / ${allC4Records.length} khách hàng`;
}

function filterC4Table() {
  const q = document.getElementById('c4SearchInput').value.trim().toLowerCase();
  if (!q) { renderC4Table(allC4Records); return; }
  const filtered = allC4Records.filter(r =>
    (r.ma_khach_hang || '').toLowerCase().includes(q) ||
    (r.ten_dang_ky_kinh_doanh || '').toLowerCase().includes(q) ||
    (r.ten_cua_hang || '').toLowerCase().includes(q) ||
    (r.giay_phep_so || '').toLowerCase().includes(q) ||
    (r.dia_chi_kinh_doanh || '').toLowerCase().includes(q)
  );
  renderC4Table(filtered);
}

/* ── Delete from list ── */
function confirmDeleteC4Record(id, name) {
  _deletePending = { type: 'c4', id, name };
  document.getElementById('modalBody').textContent =
    `Bạn có chắc muốn xóa khách hàng "${name}"? Thao tác này không thể hoàn tác.`;
  document.getElementById('confirmModal').classList.add('show');
}

/* ── Delete from detail page ── */
function confirmDeleteC4() {
  if (!c4DetailId) return;
  const title = document.getElementById('c4DetailTitle').textContent;
  confirmDeleteC4Record(c4DetailId, title);
}

/* ── Create flow ── */
function resetC4CreateForm() {
  c4PdfFile   = null;
  c4Extracted = null;
  document.getElementById('c4PdfFile').value = '';
  document.getElementById('c4DropPh').style.display    = 'block';
  document.getElementById('c4FileInfo').style.display  = 'none';
  document.getElementById('c4PdfDrop').classList.remove('has-file','drag-over');
  document.getElementById('c4ExtractBtn').disabled = true;
  document.getElementById('c4ExtractLoading').style.display = 'none';
  document.getElementById('c4Step1').style.display = 'block';
  document.getElementById('c4Step2').style.display = 'none';
  setStep('c4s', 1, 'active');
  setStep('c4s', 2, '');
  setStep('c4s', 3, '');
}

function c4BackToStep1() {
  document.getElementById('c4Step1').style.display = 'block';
  document.getElementById('c4Step2').style.display = 'none';
  setStep('c4s', 1, 'active');
  setStep('c4s', 2, '');
}

function c4DragOver(e) {
  e.preventDefault();
  document.getElementById('c4PdfDrop').classList.add('drag-over');
}
function c4DragLeave() {
  document.getElementById('c4PdfDrop').classList.remove('drag-over');
}
function c4Drop(e) {
  e.preventDefault();
  c4DragLeave();
  const f = e.dataTransfer.files[0];
  if (f) setC4File(f);
}
function onC4FileChange() {
  const f = document.getElementById('c4PdfFile').files[0];
  if (f) setC4File(f);
}
function setC4File(file) {
  c4PdfFile = file;
  document.getElementById('c4DropPh').style.display   = 'none';
  document.getElementById('c4FileInfo').style.display = 'block';
  document.getElementById('c4FileName').textContent   = file.name;
  document.getElementById('c4FileSize').textContent   = (file.size / 1024).toFixed(1) + ' KB';
  document.getElementById('c4PdfDrop').classList.add('has-file');
  document.getElementById('c4ExtractBtn').disabled = false;
}
function clearC4File(e) {
  e.stopPropagation();
  c4PdfFile = null;
  document.getElementById('c4PdfFile').value = '';
  document.getElementById('c4DropPh').style.display   = 'block';
  document.getElementById('c4FileInfo').style.display = 'none';
  document.getElementById('c4PdfDrop').classList.remove('has-file');
  document.getElementById('c4ExtractBtn').disabled = true;
}

async function runC4Extract() {
  if (!c4PdfFile) return;
  document.getElementById('c4ExtractBtn').disabled = true;
  document.getElementById('c4ExtractSpinner').style.display = 'inline-block';
  document.getElementById('c4ExtractLabel').textContent = 'Đang xử lý…';
  document.getElementById('c4ExtractLoading').style.display = 'block';

  try {
    const fd = new FormData();
    fd.append('file', c4PdfFile);
    const res = await fetch(`${API}/c4-customers/extract-preview`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    c4Extracted = data;
    populateC4Form(data);
    setStep('c4s', 1, 'done');
    setStep('c4s', 2, 'active');
    document.getElementById('c4Step1').style.display = 'none';
    document.getElementById('c4Step2').style.display = 'block';
    showToast('success', 'Trích xuất thành công', 'Kiểm tra lại thông tin trước khi lưu.');
  } catch (err) {
    showToast('error', 'Lỗi trích xuất', err.message);
  } finally {
    document.getElementById('c4ExtractBtn').disabled = false;
    document.getElementById('c4ExtractSpinner').style.display = 'none';
    document.getElementById('c4ExtractLabel').textContent = '🔍 Trích xuất thông tin';
    document.getElementById('c4ExtractLoading').style.display = 'none';
  }
}

function populateC4Form(data) {
  const ci = data.customer_info || {};
  const gp = ci.giay_phep_kinh_doanh || {};
  document.getElementById('c4FMaKH').value   = ci.ma_khach_hang || '';
  document.getElementById('c4FTenDK').value  = ci.ten_dang_ky_kinh_doanh || '';
  document.getElementById('c4FTenCH').value  = ci.ten_cua_hang || '';
  document.getElementById('c4FGPSo').value   = gp.so || '';
  document.getElementById('c4FGPNgay').value = gp.ngay_cap || '';
  document.getElementById('c4FGPNoi').value  = gp.noi_cap || '';
  document.getElementById('c4FDiaChi').value = ci.dia_chi_kinh_doanh || '';

  // Render signature preview grid
  const sigs = data.signatures || {};
  const grid = document.getElementById('c4SigPreviewGrid');
  const sigDefs = [
    { key: 'nguoi_chiu_trach_nhiem', sub: 'chu_ky_lan_1', label: 'CT — Lần 1' },
    { key: 'nguoi_chiu_trach_nhiem', sub: 'chu_ky_lan_2', label: 'CT — Lần 2' },
    { key: 'uy_quyen_1', sub: 'chu_ky_lan_1', label: 'UQ1 — Lần 1' },
    { key: 'uy_quyen_1', sub: 'chu_ky_lan_2', label: 'UQ1 — Lần 2' },
    { key: 'uy_quyen_2', sub: 'chu_ky_lan_1', label: 'UQ2 — Lần 1' },
    { key: 'uy_quyen_2', sub: 'chu_ky_lan_2', label: 'UQ2 — Lần 2' },
    { key: 'uy_quyen_3', sub: 'chu_ky_lan_1', label: 'UQ3 — Lần 1' },
    { key: 'uy_quyen_3', sub: 'chu_ky_lan_2', label: 'UQ3 — Lần 2' },
  ];
  grid.innerHTML = sigDefs.map(sd => {
    const group = sigs[sd.key];
    const sigData = group && group[sd.sub];
    const body = sigData && sigData.image_b64
      ? `<img src="data:${sigData.mime};base64,${sigData.image_b64}" alt="${sd.label}" />`
      : `<span class="sp-empty">Không có</span>`;
    return `<div class="sig-preview-card">
      <div class="sp-header">${sd.label}</div>
      <div class="sp-body">${body}</div>
    </div>`;
  }).join('');
}

async function saveC4Customer() {
  if (!c4PdfFile) return;
  document.getElementById('c4SaveBtn').disabled = true;
  document.getElementById('c4SaveSpinner').style.display = 'inline-block';
  document.getElementById('c4SaveLabel').textContent = 'Đang lưu…';

  try {
    const fd = new FormData();
    fd.append('file', c4PdfFile);
    fd.append('ma_khach_hang',          document.getElementById('c4FMaKH').value);
    fd.append('ten_dang_ky_kinh_doanh', document.getElementById('c4FTenDK').value);
    fd.append('ten_cua_hang',           document.getElementById('c4FTenCH').value);
    fd.append('giay_phep_so',           document.getElementById('c4FGPSo').value);
    fd.append('giay_phep_ngay_cap',     document.getElementById('c4FGPNgay').value);
    fd.append('giay_phep_noi_cap',      document.getElementById('c4FGPNoi').value);
    fd.append('dia_chi_kinh_doanh',     document.getElementById('c4FDiaChi').value);

    const res  = await fetch(`${API}/c4-customers`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);

    setStep('c4s', 2, 'done');
    setStep('c4s', 3, 'done');
    showToast('success', 'Lưu thành công!',
      `Khách hàng "${data.ten_dang_ky_kinh_doanh || data.id}" đã được tạo.`);
    goC4Detail(data.id);
  } catch (err) {
    showToast('error', 'Lỗi lưu', err.message);
  } finally {
    document.getElementById('c4SaveBtn').disabled = false;
    document.getElementById('c4SaveSpinner').style.display = 'none';
    document.getElementById('c4SaveLabel').textContent = '💾 Lưu khách hàng';
  }
}

/* ── Detail ── */
async function loadC4Detail(id) {
  try {
    const res  = await fetch(`${API}/c4-customers/${id}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderC4Detail(data);
  } catch (err) {
    showToast('error', 'Lỗi tải chi tiết', err.message);
  }
}

function renderC4Detail(data) {
  const val = (v) => v || '—';
  document.getElementById('c4DetailTitle').textContent    = data.ten_dang_ky_kinh_doanh || `Khách hàng #${data.id}`;
  document.getElementById('c4DetailSubtitle').textContent = `#${data.id} · ${new Date(data.created_at).toLocaleString('vi-VN')}`;
  document.getElementById('c4DetailPdfLink').href = `${API}/c4-customers/${data.id}/pdf`;

  const setText = (id, v) => {
    const el = document.getElementById(id);
    el.textContent = val(v);
    el.className = 'i-value' + (v ? '' : ' empty');
  };
  setText('c4DtMaKH',   data.ma_khach_hang);
  setText('c4DtTenDK',  data.ten_dang_ky_kinh_doanh);
  setText('c4DtTenCH',  data.ten_cua_hang);
  setText('c4DtGPSo',   data.giay_phep_so);
  setText('c4DtGPNgay', data.giay_phep_ngay_cap);
  setText('c4DtGPNoi',  data.giay_phep_noi_cap);
  setText('c4DtDiaChi', data.dia_chi_kinh_doanh);

  const sigBoxMap = {
    sig_ct_lan1:  'c4SigCT1',
    sig_ct_lan2:  'c4SigCT2',
    sig_uq1_lan1: 'c4SigUQ11',
    sig_uq1_lan2: 'c4SigUQ12',
    sig_uq2_lan1: 'c4SigUQ21',
    sig_uq2_lan2: 'c4SigUQ22',
    sig_uq3_lan1: 'c4SigUQ31',
    sig_uq3_lan2: 'c4SigUQ32',
  };
  for (const [sigKey, boxId] of Object.entries(sigBoxMap)) {
    const box = document.getElementById(boxId);
    if (data.signatures[sigKey]) {
      box.innerHTML = `<img src="${API}/c4-customers/${data.id}/signatures/${sigKey}?t=${Date.now()}"
        alt="${sigKey}" style="max-width:100%;max-height:120px;object-fit:contain" />`;
    } else {
      box.innerHTML = `<span class="no-sig">Không có chữ ký</span>`;
    }
  }
}

/* ── Delete (unified handler extension) ── */
const _origModalConfirm = document.getElementById('modalConfirmBtn').onclick;
document.getElementById('modalConfirmBtn').onclick = async () => {
  if (_deletePending && typeof _deletePending === 'object' && _deletePending.type === 'c4') {
    const { id, name } = _deletePending;
    closeModal();
    try {
      const res = await fetch(`${API}/c4-customers/${id}`, { method: 'DELETE' });
      if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
      showToast('success', 'Đã xóa', `Khách hàng "${name}" đã được xóa.`);
      goC4List();
    } catch (err) {
      showToast('error', 'Lỗi xóa', err.message);
    }
  } else if (_deletePending) {
    // original employee delete
    const code = _deletePending;
    closeModal();
    try {
      const res = await fetch(`${API}/employees/${encodeURIComponent(code)}`, { method: 'DELETE' });
      if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
      showToast('success', 'Đã xóa', `Nhân viên ${code} đã được xóa.`);
      loadList();
    } catch (err) {
      showToast('error', 'Lỗi xóa', err.message);
    }
  }
};
