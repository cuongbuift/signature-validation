/* ═══════════════════════════════════════════
   CHECK SIGNATURE MODAL
═══════════════════════════════════════════ */
let _checkCode = null;
let _checkFile = null;

function openCheckModal(code, name) {
  _checkCode = code;
  _checkFile = null;
  document.getElementById('checkSubtitle').textContent = `${name} — ${code}`;
  document.getElementById('checkDrop').classList.remove('has-file','drag-over');
  document.getElementById('checkFile').value = '';
  document.getElementById('checkImg').src = '';
  document.getElementById('checkResults').style.display = 'none';
  document.getElementById('scoreCards').innerHTML = '';
  document.getElementById('scoreSummary').innerHTML = '';
  document.getElementById('checkSubmitBtn').disabled = true;
  document.getElementById('checkSubmitLabel').textContent = 'Kiểm tra';
  document.getElementById('checkSpinner').style.display = 'none';
  document.getElementById('checkModal').classList.add('show');
}

function closeCheckModal() {
  document.getElementById('checkModal').classList.remove('show');
  if (_checkFile) { URL.revokeObjectURL(document.getElementById('checkImg').src); }
  _checkCode = null; _checkFile = null;
}

document.getElementById('checkModal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeCheckModal();
});

function setCheckFile(file) {
  if (!file) return;
  _checkFile = file;
  const url = URL.createObjectURL(file);
  document.getElementById('checkImg').src = url;
  document.getElementById('checkDrop').classList.add('has-file');
  document.getElementById('checkResults').style.display = 'none';
  document.getElementById('checkSubmitBtn').disabled = false;
}

function onCheckFileChange() {
  const f = document.getElementById('checkFile').files[0];
  if (f) setCheckFile(f);
}

function clearCheckFile(e) {
  e.stopPropagation();
  _checkFile = null;
  document.getElementById('checkFile').value = '';
  URL.revokeObjectURL(document.getElementById('checkImg').src);
  document.getElementById('checkImg').src = '';
  document.getElementById('checkDrop').classList.remove('has-file');
  document.getElementById('checkResults').style.display = 'none';
  document.getElementById('checkSubmitBtn').disabled = true;
}

function checkDragOver(ev) {
  ev.preventDefault();
  document.getElementById('checkDrop').classList.add('drag-over');
}
function checkDragLeave() {
  document.getElementById('checkDrop').classList.remove('drag-over');
}
function checkDrop_(ev) {
  ev.preventDefault();
  document.getElementById('checkDrop').classList.remove('drag-over');
  const f = ev.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setCheckFile(f);
  else showToast('error','Định dạng không hỗ trợ','Chỉ chấp nhận JPEG, PNG, WebP, TIFF.');
}

async function submitCheck() {
  if (!_checkFile || !_checkCode) return;

  document.getElementById('checkSubmitBtn').disabled = true;
  document.getElementById('checkSpinner').style.display = 'block';
  document.getElementById('checkSubmitLabel').textContent = 'Đang xử lý…';
  document.getElementById('checkResults').style.display = 'none';

  try {
    const fd = new FormData();
    fd.append('employee_code', _checkCode);
    fd.append('file', _checkFile);

    const res = await fetch(`${API}/validate`, { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({}));
      throw new Error(err.detail || `Lỗi ${res.status}`);
    }
    const data = await res.json();
    renderCheckResults(data);
  } catch (err) {
    showToast('error', 'Kiểm tra thất bại', err.message);
  } finally {
    document.getElementById('checkSubmitBtn').disabled = false;
    document.getElementById('checkSpinner').style.display = 'none';
    document.getElementById('checkSubmitLabel').textContent = 'Kiểm tra lại';
  }
}

function renderCheckResults(data) {
  const perRef = data.detail?.per_reference || [];
  const threshold = data.threshold_used;

  const cards = perRef.map((ref, i) => {
    const overall = ref.overall;
    const isValid = overall >= threshold;
    const pct = (v) => (v * 100).toFixed(1) + '%';
    const imgUrl = ref.signature_id
      ? `${API}/employees/${data.employee_code}/signatures/${ref.signature_id}/image`
      : null;
    const imgTag = imgUrl
      ? `<img class="score-ref-img" src="${imgUrl}" alt="Chữ ký mẫu ${i + 1}" />`
      : '';
    return `
      <div class="score-card ${isValid ? 'valid' : 'invalid'}">
        <div class="score-card-title">
          Chữ ký mẫu ${i + 1}
          <span class="verdict ${isValid ? 'ok' : 'no'}">${isValid ? '✓ Khớp' : '✗ Không khớp'}</span>
        </div>
        <div class="score-card-body">
          ${imgTag}
          <div class="score-metrics">
            <div class="score-metric">
              <div class="m-val">${pct(overall)}</div>
              <div class="m-lbl">Tổng hợp</div>
            </div>
            <div class="score-metric" title="Siamese Neural Network — học trực tiếp từ chữ ký mẫu">
              <div class="m-val" style="color:#6b21a8">${pct(ref.siamese ?? 0)}</div>
              <div class="m-lbl">Siamese</div>
            </div>
            <div class="score-metric" title="EfficientNet-B0 deep feature cosine similarity">
              <div class="m-val" style="color:#7e22ce">${pct(ref.deep ?? 0)}</div>
              <div class="m-lbl">Deep CNN</div>
            </div>
            <div class="score-metric" title="SigNet (sigver) — pretrained signature verification model">
              <div class="m-val" style="color:#0369a1">${pct(ref.sigver ?? 0)}</div>
              <div class="m-lbl">SigVer</div>
            </div>
            <div class="score-metric">
              <div class="m-val">${pct(ref.ssim)}</div>
              <div class="m-lbl">SSIM</div>
            </div>
            <div class="score-metric">
              <div class="m-val">${pct(ref.orb)}</div>
              <div class="m-lbl">ORB</div>
            </div>
            <div class="score-metric">
              <div class="m-val">${pct(ref.contour)}</div>
              <div class="m-lbl">Contour</div>
            </div>
          </div>
        </div>
      </div>`;
  });

  document.getElementById('scoreCards').innerHTML = cards.join('');

  const overallPct = (data.overall_score * 100).toFixed(1) + '%';
  const threshPct  = (threshold * 100).toFixed(1) + '%';
  const bestValid  = data.is_valid;
  document.getElementById('scoreSummary').innerHTML = `
    <span>Ngưỡng chấp nhận: <strong>${threshPct}</strong></span>
    <span>Điểm tốt nhất: <strong style="color:${bestValid ? 'var(--green)' : 'var(--red)'}">${overallPct}</strong>
      — <strong style="color:${bestValid ? 'var(--green)' : 'var(--red)'}">${bestValid ? 'HỢP LỆ' : 'KHÔNG HỢP LỆ'}</strong>
    </span>`;

  document.getElementById('checkResults').style.display = 'block';
}
