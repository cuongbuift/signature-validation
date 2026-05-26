/* ═══════════════════════════════════════════
   LIST PAGE
═══════════════════════════════════════════ */
async function loadList() {
  renderSkeleton();
  try {
    const res = await fetch(`${API}/employees?limit=200`);
    allEmployees = await res.json();
    renderTable(allEmployees);
  } catch {
    showToast('error', 'Lỗi tải dữ liệu', 'Không thể kết nối tới server.');
    renderTable([]);
  }
}

function renderSkeleton() {
  const tbody = document.getElementById('empTbody');
  tbody.innerHTML = Array(4).fill(0).map(() => `
    <tr class="skel-row">
      <td><div class="skeleton" style="width:20px"></div></td>
      <td><div class="skeleton" style="width:70px"></div></td>
      <td><div class="skeleton" style="width:140px"></div></td>
      <td><div class="skeleton" style="width:60px"></div></td>
      <td><div class="skeleton" style="width:90px"></div></td>
      <td><div class="skeleton" style="width:100px"></div></td>
    </tr>`).join('');
  document.getElementById('tableMeta').textContent = '';
}

function renderTable(rows) {
  const tbody = document.getElementById('empTbody');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6">
      <div class="empty-state">
        <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"/></svg>
        <p>Chưa có nhân viên nào. <span class="crumb-link" onclick="goCreate()">Thêm ngay</span></p>
      </div>
    </td></tr>`;
    document.getElementById('tableMeta').textContent = '';
    return;
  }

  tbody.innerHTML = rows.map((e, i) => {
    const hasSig = e.signature_count >= 2;
    const badge  = hasSig
      ? `<span class="badge badge-sig">✓ ${e.signature_count} chữ ký</span>`
      : `<span class="badge badge-nosig">⚠ ${e.signature_count}/2</span>`;
    const date = new Date(e.created_at).toLocaleDateString('vi-VN');
    return `<tr>
      <td style="color:var(--gray-400);font-size:.8rem">${i + 1}</td>
      <td><strong>${esc(e.employee_code)}</strong></td>
      <td>${esc(e.full_name)}</td>
      <td>${badge}</td>
      <td style="color:var(--gray-500);font-size:.82rem">${date}</td>
      <td>
        <div class="actions-cell">
          <button class="btn-icon check" onclick="openCheckModal('${esc(e.employee_code)}','${esc(e.full_name)}')" ${hasSig ? '' : 'disabled title="Cần có đủ 2 chữ ký mẫu"'}>
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            Kiểm tra
          </button>
          <button class="btn-icon edit" onclick="goEdit('${esc(e.employee_code)}')">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125"/></svg>
            Sửa
          </button>
          <button class="btn-icon delete" onclick="confirmDelete('${esc(e.employee_code)}','${esc(e.full_name)}')">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/></svg>
            Xóa
          </button>
        </div>
      </td>
    </tr>`;
  }).join('');

  document.getElementById('tableMeta').textContent =
    `Hiển thị ${rows.length} / ${allEmployees.length} nhân viên`;
}

function filterTable() {
  const q = document.getElementById('searchInput').value.trim().toLowerCase();
  if (!q) { renderTable(allEmployees); return; }
  const filtered = allEmployees.filter(e =>
    e.employee_code.toLowerCase().includes(q) ||
    e.full_name.toLowerCase().includes(q)
  );
  renderTable(filtered);
}

/* ═══════════════════════════════════════════
   DELETE
═══════════════════════════════════════════ */
let _deletePending = null;

function confirmDelete(code, name) {
  _deletePending = code;
  document.getElementById('modalBody').textContent =
    `Bạn có chắc muốn xóa nhân viên "${name}" (${code})? Hành động này không thể hoàn tác.`;
  document.getElementById('confirmModal').classList.add('show');
}

function closeModal() {
  document.getElementById('confirmModal').classList.remove('show');
  _deletePending = null;
}

// Close modal on backdrop click
document.getElementById('confirmModal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

/* ═══════════════════════════════════════════
   CREATE FORM
═══════════════════════════════════════════ */
function resetCreateForm() {
  document.getElementById('cCode').value = '';
  document.getElementById('cName').value = '';
  ['cCode','cName'].forEach(id => {
    document.getElementById(id).classList.remove('err');
    document.getElementById(id + 'Err').style.display = 'none';
  });
  for (const s of [1,2]) removeFile({ stopPropagation:()=>{} }, 'c', s);
  ['cSigErr1','cSigErr2'].forEach(id => document.getElementById(id).style.display = 'none');
  setStep('cs',1,'active'); setStep('cs',2,''); setStep('cs',3,'');
}

function syncCreateSteps() {
  const code = document.getElementById('cCode').value.trim();
  const name = document.getElementById('cName').value.trim();
  const hasInfo = code && name;
  const hasSigs = files.c[1] && files.c[2];
  setStep('cs',1, hasInfo ? 'done':'active');
  setStep('cs',2, hasSigs ? 'done' : (hasInfo ? 'active':''));
  setStep('cs',3,'');
}

document.getElementById('createForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const code = document.getElementById('cCode').value.trim();
  const name = document.getElementById('cName').value.trim();
  let ok = true;

  if (!code) { setFieldErr('cCode','cCodeErr',true); ok=false; } else setFieldErr('cCode','cCodeErr',false);
  if (!name) { setFieldErr('cName','cNameErr',true); ok=false; } else setFieldErr('cName','cNameErr',false);
  for (const s of [1,2]) {
    if (!files.c[s]) {
      document.getElementById('cSlot'+s).classList.add('err-slot');
      document.getElementById('cSigErr'+s).style.display = 'block';
      ok = false;
    }
  }
  if (!ok) return;

  setLoading('c', true);
  try {
    const empRes = await fetch(`${API}/employees`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ employee_code: code, full_name: name }),
    });
    if (!empRes.ok) {
      const err = await empRes.json().catch(()=>({}));
      throw new Error(err.detail || `Lỗi tạo nhân viên (${empRes.status})`);
    }

    for (const s of [1,2]) {
      const fd = new FormData();
      fd.append('file', files.c[s]);
      const r = await fetch(`${API}/employees/${encodeURIComponent(code)}/signatures`, { method:'POST', body:fd });
      if (!r.ok) {
        const err = await r.json().catch(()=>({}));
        throw new Error(err.detail || `Lỗi upload chữ ký ${s}`);
      }
    }

    setStep('cs',3,'done');
    showToast('success','Tạo thành công!', `Nhân viên ${name} (${code}) đã được thêm.`);
    goList();
  } catch(err) {
    showToast('error','Có lỗi xảy ra', err.message);
  } finally {
    setLoading('c', false);
  }
});

/* ═══════════════════════════════════════════
   EDIT FORM
═══════════════════════════════════════════ */
async function loadEditForm(code) {
  // reset file slots
  for (const s of [1,2]) removeFile({ stopPropagation:()=>{} }, 'e', s);

  document.getElementById('eCode').value = code;
  document.getElementById('eName').value = '';
  document.getElementById('eNameErr').style.display = 'none';
  document.getElementById('eName').classList.remove('err');
  document.getElementById('editSubtitle').textContent = `Mã NV: ${code}`;
  setStep('es',1,'active'); setStep('es',2,''); setStep('es',3,'');

  // reset existing indicators
  for (const s of [1,2]) {
    const existEl = document.getElementById('eExist'+s);
    const imgEl   = document.getElementById('eExistImg'+s);
    existEl.style.display = 'none';
    imgEl.src = '';
    document.getElementById('eSlot'+s).classList.remove('existing');
    document.getElementById('eSigErr'+s).style.display = 'none';
    document.getElementById('eSigErr'+s).textContent = '';
  }

  try {
    const [empRes, sigRes] = await Promise.all([
      fetch(`${API}/employees/${encodeURIComponent(code)}`),
      fetch(`${API}/employees/${encodeURIComponent(code)}/signatures`),
    ]);
    const emp  = await empRes.json();
    editSigs   = await sigRes.json();   // [{id, order, …}, …]

    document.getElementById('eName').value = emp.full_name;

    // Show existing sig images
    for (const s of [1,2]) {
      const sig = editSigs.find(sg => sg.order === s);
      if (sig) {
        const imgEl = document.getElementById('eExistImg'+s);
        // Load actual image from server; add cache-buster to avoid stale after replace
        imgEl.src = `${API}/employees/${encodeURIComponent(code)}/signatures/${sig.id}/image?t=${Date.now()}`;
        document.getElementById('eExist'+s).style.display = 'block';
        document.getElementById('eSlot'+s).classList.add('existing');
      }
    }
    syncEditSteps();
  } catch {
    showToast('error','Lỗi tải dữ liệu','Không thể tải thông tin nhân viên.');
  }
}

function syncEditSteps() {
  const name = document.getElementById('eName').value.trim();
  setStep('es',1, name ? 'done':'active');
  setStep('es',2, 'active');
  setStep('es',3,'');
}

document.getElementById('editForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const code = editCode;
  const name = document.getElementById('eName').value.trim();
  if (!name) { setFieldErr('eName','eNameErr',true); return; }
  setFieldErr('eName','eNameErr',false);

  // Determine which slots need uploading
  const slotsToUpload = [1,2].filter(s => files.e[s]);

  // For slots where there's an existing sig and a NEW file: delete old first
  // For slots where there's no existing sig and no new file: warn if < 2 total
  const existingOrders = editSigs.map(sg => sg.order);
  const willHaveSig = [1,2].map(s =>
    files.e[s] !== null || existingOrders.includes(s)
  );
  if (!willHaveSig[0] || !willHaveSig[1]) {
    for (const s of [1,2]) {
      if (!willHaveSig[s-1]) {
        document.getElementById('eSlot'+s).classList.add('err-slot');
        document.getElementById('eSigErr'+s).textContent = `Vui lòng chọn chữ ký lần ${s}.`;
        document.getElementById('eSigErr'+s).style.display = 'block';
      }
    }
    return;
  }

  setLoading('e', true);
  try {
    // 1. Update name
    await fetch(`${API}/employees/${encodeURIComponent(code)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_name: name }),
    }).then(r => { if (!r.ok) throw new Error(`Lỗi cập nhật tên (${r.status})`); });

    // 2. For each new file: delete existing same-order sig then upload
    for (const s of slotsToUpload) {
      const oldSig = editSigs.find(sg => sg.order === s);
      if (oldSig) {
        await fetch(`${API}/employees/${encodeURIComponent(code)}/signatures/${oldSig.id}`, {
          method: 'DELETE',
        });
      }
      const fd = new FormData();
      fd.append('file', files.e[s]);
      const r = await fetch(`${API}/employees/${encodeURIComponent(code)}/signatures`, { method:'POST', body:fd });
      if (!r.ok) {
        const err = await r.json().catch(()=>({}));
        throw new Error(err.detail || `Lỗi upload chữ ký ${s}`);
      }
    }

    setStep('es',3,'done');
    showToast('success','Lưu thành công!', `Thông tin nhân viên ${code} đã được cập nhật.`);
    goList();
  } catch(err) {
    showToast('error','Có lỗi xảy ra', err.message);
  } finally {
    setLoading('e', false);
  }
});

/* ═══════════════════════════════════════════
   FILE HANDLING (shared)
═══════════════════════════════════════════ */
function onFileChange(ctx, slot) {
  const input = document.getElementById(ctx + 'File' + slot);
  setFile(ctx, slot, input.files[0] || null);
}

function setFile(ctx, slot, file) {
  if (!file) return;
  files[ctx][slot] = file;

  const url = URL.createObjectURL(file);
  document.getElementById(ctx + 'Img' + slot).src = url;
  document.getElementById(ctx + 'FName' + slot).textContent = file.name;

  const slotEl = document.getElementById(ctx + 'Slot' + slot);
  slotEl.classList.add('has-file');
  slotEl.classList.remove('err-slot', 'existing');

  if (ctx === 'e') {
    document.getElementById('eExist'+slot).style.display = 'none';
  }

  // hide error
  const errEl = document.getElementById(ctx === 'c' ? `cSigErr${slot}` : `eSigErr${slot}`);
  if (errEl) { errEl.style.display = 'none'; errEl.textContent = ''; }

  ctx === 'c' ? syncCreateSteps() : syncEditSteps();
}

function removeFile(e, ctx, slot) {
  e.stopPropagation();
  files[ctx][slot] = null;
  const input = document.getElementById(ctx + 'File' + slot);
  if (input) input.value = '';
  const img = document.getElementById(ctx + 'Img' + slot);
  if (img) { URL.revokeObjectURL(img.src); img.src = ''; }

  const slotEl = document.getElementById(ctx + 'Slot' + slot);
  slotEl.classList.remove('has-file');

  // restore existing indicator on edit page
  if (ctx === 'e') {
    const exists = editSigs.some(sg => sg.order === slot);
    if (exists) {
      document.getElementById('eExist'+slot).style.display = 'flex';
      slotEl.classList.add('existing');
    }
    syncEditSteps();
  } else {
    syncCreateSteps();
  }
}

function onDragOver(ev, ctx, slot) {
  ev.preventDefault();
  document.getElementById(ctx+'Slot'+slot).classList.add('drag-over');
}
function onDragLeave(ctx, slot) {
  document.getElementById(ctx+'Slot'+slot).classList.remove('drag-over');
}
function onDrop(ev, ctx, slot) {
  ev.preventDefault();
  document.getElementById(ctx+'Slot'+slot).classList.remove('drag-over');
  const file = ev.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    setFile(ctx, slot, file);
  } else {
    showToast('error','Định dạng không hỗ trợ','Chỉ chấp nhận JPEG, PNG, WebP, TIFF.');
  }
}
