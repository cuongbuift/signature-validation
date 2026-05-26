/* ═══════════════════════════════════════════
   DO VALIDATE MODAL
═══════════════════════════════════════════ */
let _doValidateId   = null;
let _doValidateFiles = [];

function openDoValidateModal(id, name) {
  _doValidateId    = id;
  _doValidateFiles = [];
  document.getElementById('doValidateSubtitle').textContent = `${name} — #${id}`;
  document.getElementById('doFiles').value = '';
  renderDoFileList();
  document.getElementById('doValidateResults').style.display = 'none';
  document.getElementById('doValidateBtn').disabled = true;
  document.getElementById('doValidateLabel').textContent = '🔍 Xác thực';
  document.getElementById('doValidateSpinner').style.display = 'none';
  document.getElementById('doValidateModal').classList.add('show');
}

function openDoValidateModalFromDetail() {
  if (!c4DetailId) return;
  const name = document.getElementById('c4DetailTitle').textContent;
  openDoValidateModal(c4DetailId, name);
}

function closeDoValidateModal() {
  document.getElementById('doValidateModal').classList.remove('show');
  _doValidateId    = null;
  _doValidateFiles = [];
}

document.getElementById('doValidateModal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeDoValidateModal();
});

function doDragOver(e) {
  e.preventDefault();
  document.getElementById('doDropZone').classList.add('drag-over');
}
function doDragLeave() {
  document.getElementById('doDropZone').classList.remove('drag-over');
}
function doDrop_(e) {
  e.preventDefault();
  doDragLeave();
  addDoFiles(e.dataTransfer.files);
}
function onDoFilesChange() {
  addDoFiles(document.getElementById('doFiles').files);
}

function addDoFiles(fileList) {
  for (const f of fileList) {
    if ((f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')) &&
        !_doValidateFiles.some(x => x.name === f.name && x.size === f.size)) {
      _doValidateFiles.push(f);
    }
  }
  renderDoFileList();
  document.getElementById('doFiles').value = '';
}

function removeDoFile(idx) {
  _doValidateFiles.splice(idx, 1);
  renderDoFileList();
}

function renderDoFileList() {
  const el = document.getElementById('doFileList');
  if (!_doValidateFiles.length) {
    el.style.display = 'none';
    document.getElementById('doValidateBtn').disabled = true;
    return;
  }
  el.style.cssText = 'display:flex;margin-top:12px;flex-wrap:wrap;gap:8px';
  el.innerHTML = _doValidateFiles.map((f, i) => `
    <div class="do-file-tag">
      📄 ${esc(f.name)} <span style="color:var(--gray-400)">(${(f.size/1024).toFixed(0)}KB)</span>
      <button class="rm" onclick="removeDoFile(${i})">×</button>
    </div>`).join('');
  document.getElementById('doValidateBtn').disabled = false;
}

async function submitDoValidate() {
  if (!_doValidateId || !_doValidateFiles.length) return;

  document.getElementById('doValidateBtn').disabled = true;
  document.getElementById('doValidateSpinner').style.display = 'block';
  document.getElementById('doValidateLabel').textContent = 'Đang xử lý…';
  document.getElementById('doValidateResults').style.display = 'none';

  try {
    const fd = new FormData();
    for (const f of _doValidateFiles) fd.append('files', f);

    const res  = await fetch(`${API}/c4-customers/${_doValidateId}/validate-do`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    renderDoValidateResults(data);
  } catch (err) {
    showToast('error', 'Lỗi xác thực', err.message);
  } finally {
    document.getElementById('doValidateBtn').disabled = false;
    document.getElementById('doValidateSpinner').style.display = 'none';
    document.getElementById('doValidateLabel').textContent = '🔍 Xác thực lại';
  }
}

function renderDoValidateResults(data) {
  const pct = v => (v * 100).toFixed(1) + '%';
  const thresh = data.threshold;

  let html = `<div class="do-result-banner ${data.do_valid ? 'valid' : 'invalid'}">
    ${data.do_valid
      ? '✅ KẾT LUẬN: HỢP LỆ — Ít nhất một chữ ký DO khớp với chữ ký mẫu của khách hàng'
      : '❌ KẾT LUẬN: KHÔNG HỢP LỆ — Không có chữ ký DO nào khớp với chữ ký mẫu'}
    <span style="margin-left:auto;font-weight:400;font-size:.82rem;opacity:.8">Ngưỡng: ${pct(thresh)}</span>
  </div>`;

  for (const doItem of data.dos) {
    const headerCls = doItem.is_valid ? 'color:#166534' : 'color:var(--gray-700)';
    const badge = doItem.error
      ? `<span style="font-size:.75rem;color:var(--red);background:#fee2e2;padding:2px 8px;border-radius:99px">Lỗi</span>`
      : doItem.is_valid
        ? `<span style="font-size:.75rem;color:#166534;background:#dcfce7;padding:2px 8px;border-radius:99px;font-weight:700">✓ Hợp lệ</span>`
        : `<span style="font-size:.75rem;color:#991b1b;background:#fee2e2;padding:2px 8px;border-radius:99px;font-weight:700">✗ Không hợp lệ</span>`;

    html += `<div class="do-file-block">
      <div class="do-file-block-header">
        <span style="${headerCls}">📄 ${esc(doItem.filename)}</span>
        <span style="display:flex;align-items:center;gap:8px">
          <span style="font-size:.75rem;color:var(--gray-500)">${doItem.extracted_count} chữ ký trích xuất</span>
          ${badge}
        </span>
      </div>`;

    if (doItem.error) {
      html += `<div style="padding:16px;font-size:.85rem;color:var(--red)">${esc(doItem.error)}</div>`;
    } else if (!doItem.extracted_signatures.length) {
      html += `<div style="padding:16px;font-size:.85rem;color:var(--gray-400);text-align:center">Không tìm thấy chữ ký trong file này</div>`;
    } else {
      for (const sig of doItem.extracted_signatures) {
        const sigBadge = sig.is_valid
          ? `<span style="color:#166534;font-size:.72rem;background:#dcfce7;padding:1px 7px;border-radius:99px;font-weight:700">✓ Khớp (${pct(sig.best_score)})</span>`
          : `<span style="color:#991b1b;font-size:.72rem;background:#fee2e2;padding:1px 7px;border-radius:99px">✗ Không khớp (${pct(sig.best_score)})</span>`;

        html += `<div class="do-sig-block">
          <div class="do-sig-block-title">
            Chữ ký trang ${sig.page} ${sigBadge}
          </div>
          <div class="do-sig-row">
            <div class="do-extracted-img">
              <img src="data:image/png;base64,${sig.image_b64}" alt="Chữ ký DO trang ${sig.page}" />
              <div class="label">Chữ ký từ DO</div>
            </div>
            <div class="do-comparison-grid">`;

        for (const cmp of sig.comparisons) {
          const validCls = cmp.is_valid ? 'valid' : '';
          const scoreColor = cmp.is_valid ? '#166534' : (cmp.overall_score >= thresh * 0.8 ? '#92400e' : '#9ca3af');
          html += `<div class="do-cmp-card ${validCls}">
            <div class="do-cmp-card-header">
              <span>${esc(cmp.sig_label)}</span>
              ${cmp.is_valid ? '<span>✓</span>' : ''}
            </div>
            <div class="do-cmp-card-img">
              <img src="${API}/c4-customers/${_doValidateId}/signatures/${cmp.sig_key}?t=${Date.now()}"
                   alt="${esc(cmp.sig_label)}"
                   onerror="this.parentElement.innerHTML='<span style=\\'font-size:.65rem;color:var(--gray-400)\\'>Không có ảnh</span>'" />
            </div>
            <div class="do-cmp-card-score" style="color:${scoreColor}">
              ${cmp.error ? 'Lỗi' : pct(cmp.overall_score || 0)}
            </div>
          </div>`;
        }

        html += `    </div>
          </div>
        </div>`;
      }
    }
    html += '</div>';
  }

  const resEl = document.getElementById('doValidateResults');
  resEl.innerHTML = html;
  resEl.style.display = 'block';
}

/* ═══════════════════════════════════════════
   CHECK DO BATCH MODAL (from customer list)
═══════════════════════════════════════════ */
let _batchDoFiles = [];

function openCheckDoBatchModal() {
  _batchDoFiles = [];
  document.getElementById('batchDoFiles').value = '';
  renderBatchDoFileList();
  document.getElementById('batchDoResults').style.display = 'none';
  document.getElementById('batchDoLabel').textContent = '🔍 Kiểm tra';
  document.getElementById('batchDoSpinner').style.display = 'none';
  document.getElementById('batchDoCheckBtn').disabled = true;
  document.getElementById('checkDoBatchModal').classList.add('show');
}

function closeCheckDoBatchModal() {
  document.getElementById('checkDoBatchModal').classList.remove('show');
  _batchDoFiles = [];
}

document.getElementById('checkDoBatchModal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeCheckDoBatchModal();
});

function batchDoDragOver(e) {
  e.preventDefault();
  document.getElementById('batchDoDropZone').classList.add('drag-over');
}
function batchDoDragLeave() {
  document.getElementById('batchDoDropZone').classList.remove('drag-over');
}
function batchDoDrop(e) {
  e.preventDefault();
  batchDoDragLeave();
  addBatchDoFiles(e.dataTransfer.files);
}
function onBatchDoFilesChange() {
  addBatchDoFiles(document.getElementById('batchDoFiles').files);
}

function addBatchDoFiles(fileList) {
  for (const f of fileList) {
    if ((f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')) &&
        !_batchDoFiles.some(x => x.name === f.name && x.size === f.size)) {
      _batchDoFiles.push(f);
    }
  }
  renderBatchDoFileList();
  document.getElementById('batchDoFiles').value = '';
}

function removeBatchDoFile(idx) {
  _batchDoFiles.splice(idx, 1);
  renderBatchDoFileList();
}

function renderBatchDoFileList() {
  const el = document.getElementById('batchDoFileList');
  if (!_batchDoFiles.length) {
    el.style.display = 'none';
    document.getElementById('batchDoCheckBtn').disabled = true;
    return;
  }
  el.style.cssText = 'display:flex;margin-top:12px;flex-wrap:wrap;gap:8px';
  el.innerHTML = _batchDoFiles.map((f, i) => `
    <div class="do-file-tag">
      📄 ${esc(f.name)} <span style="color:var(--gray-400)">(${(f.size/1024).toFixed(0)}KB)</span>
      <button class="rm" onclick="removeBatchDoFile(${i})">×</button>
    </div>`).join('');
  document.getElementById('batchDoCheckBtn').disabled = false;
}

async function submitBatchDoCheck() {
  if (!_batchDoFiles.length) return;

  document.getElementById('batchDoCheckBtn').disabled = true;
  document.getElementById('batchDoSpinner').style.display = 'block';
  document.getElementById('batchDoLabel').textContent = 'Đang xử lý…';
  document.getElementById('batchDoResults').style.display = 'none';

  try {
    const fd = new FormData();
    for (const f of _batchDoFiles) fd.append('files', f);

    const res  = await fetch(`${API}/c4-customers/check-do-batch`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    renderBatchDoResults(data);
  } catch (err) {
    showToast('error', 'Lỗi kiểm tra DO', err.message);
  } finally {
    document.getElementById('batchDoCheckBtn').disabled = false;
    document.getElementById('batchDoSpinner').style.display = 'none';
    document.getElementById('batchDoLabel').textContent = '🔍 Kiểm tra lại';
  }
}

function renderBatchDoResults(data) {
  const pct = v => (v * 100).toFixed(1) + '%';
  const thresh = data.threshold;

  let html = `<div class="do-result-banner ${data.overall_valid ? 'valid' : 'invalid'}">
    ${data.overall_valid
      ? '✅ KẾT LUẬN: CÓ FILE HỢP LỆ — Ít nhất một DO có chữ ký khớp với khách hàng'
      : '❌ KẾT LUẬN: KHÔNG HỢP LỆ — Không có chữ ký DO nào khớp với chữ ký mẫu'}
    <span style="margin-left:auto;font-weight:400;font-size:.82rem;opacity:.8">Ngưỡng: ${pct(thresh)}</span>
  </div>`;

  for (const item of data.results) {
    const badge = item.error
      ? `<span style="font-size:.75rem;color:var(--red);background:#fee2e2;padding:2px 8px;border-radius:99px">Lỗi</span>`
      : item.is_valid
        ? `<span style="font-size:.75rem;color:#166534;background:#dcfce7;padding:2px 8px;border-radius:99px;font-weight:700">✓ Hợp lệ</span>`
        : `<span style="font-size:.75rem;color:#991b1b;background:#fee2e2;padding:2px 8px;border-radius:99px;font-weight:700">✗ Không hợp lệ</span>`;

    const customerInfo = item.customer_found
      ? `<span style="font-size:.75rem;color:var(--primary);background:#ede9fe;padding:2px 8px;border-radius:99px">
           Mã KH: ${esc(item.ma_khach_hang)} — ${esc(item.customer_name)}
         </span>`
      : item.ma_khach_hang
        ? `<span style="font-size:.75rem;color:#92400e;background:#fef3c7;padding:2px 8px;border-radius:99px">
             Mã KH: ${esc(item.ma_khach_hang)} — Không tìm thấy trong hệ thống
           </span>`
        : `<span style="font-size:.75rem;color:var(--gray-500);background:var(--gray-100);padding:2px 8px;border-radius:99px">
             Không đọc được mã KH
           </span>`;

    html += `<div class="do-file-block">
      <div class="do-file-block-header">
        <div style="display:flex;flex-direction:column;gap:4px">
          <span style="color:var(--gray-700)">📄 ${esc(item.filename)}${item.do_so ? ` &nbsp;·&nbsp; <span style="color:var(--gray-500);font-weight:400">DO: ${esc(item.do_so)}</span>` : ''}</span>
          <div style="display:flex;gap:6px;flex-wrap:wrap">${customerInfo}</div>
        </div>
        <span style="display:flex;align-items:center;gap:8px;flex-shrink:0">
          ${!item.error ? `<span style="font-size:.75rem;color:var(--gray-500)">${item.extracted_count || 0} chữ ký</span>` : ''}
          ${badge}
        </span>
      </div>`;

    if (item.error) {
      html += `<div style="padding:12px 16px;font-size:.85rem;color:var(--red)">${esc(item.error)}</div>`;
    } else if (!item.extracted_signatures || !item.extracted_signatures.length) {
      html += `<div style="padding:12px 16px;font-size:.85rem;color:var(--gray-400);text-align:center">Không tìm thấy chữ ký trong file này</div>`;
    } else {
      for (const sig of item.extracted_signatures) {
        const sigBadge = sig.is_valid
          ? `<span style="color:#166534;font-size:.72rem;background:#dcfce7;padding:1px 7px;border-radius:99px;font-weight:700">✓ Khớp (${pct(sig.best_score)})</span>`
          : `<span style="color:#991b1b;font-size:.72rem;background:#fee2e2;padding:1px 7px;border-radius:99px">✗ Không khớp (${pct(sig.best_score)})</span>`;

        html += `<div class="do-sig-block">
          <div class="do-sig-block-title">Chữ ký trang ${sig.page} ${sigBadge}</div>
          <div class="do-sig-row">
            <div class="do-extracted-img">
              <img src="data:image/png;base64,${sig.image_b64}" alt="Chữ ký DO" />
              <div class="label">Chữ ký từ DO</div>
            </div>
            <div class="do-comparison-grid">`;

        for (const cmp of sig.comparisons) {
          const scoreColor = cmp.is_valid ? '#166534' : (cmp.overall_score >= thresh * 0.8 ? '#92400e' : '#9ca3af');
          html += `<div class="do-cmp-card ${cmp.is_valid ? 'valid' : ''}">
            <div class="do-cmp-card-header">
              <span>${esc(cmp.sig_label)}</span>
              ${cmp.is_valid ? '<span>✓</span>' : ''}
            </div>
            <div class="do-cmp-card-img">
              <img src="${API}/c4-customers/${item.customer_id}/signatures/${cmp.sig_key}?t=${Date.now()}"
                   alt="${esc(cmp.sig_label)}"
                   onerror="this.parentElement.innerHTML='<span style=\\'font-size:.65rem;color:var(--gray-400)\\'>Không có ảnh</span>'" />
            </div>
            <div class="do-cmp-card-score" style="color:${scoreColor}">
              ${cmp.error ? 'Lỗi' : pct(cmp.overall_score || 0)}
            </div>
          </div>`;
        }

        html += `    </div></div></div>`;
      }
    }
    html += '</div>';
  }

  const resEl = document.getElementById('batchDoResults');
  resEl.innerHTML = html;
  resEl.style.display = 'block';
}
