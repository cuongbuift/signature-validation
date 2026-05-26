/* ═══════════════════════════════════════════
   EXTRACT SIGNATURES PAGE
═══════════════════════════════════════════ */
let extractedSignatures = [];

function goExtract() {
  showPage('extract');
  setBreadcrumb([{ label: 'Trích xuất chữ ký từ DO' }]);
}

function extractDragOver(e) {
  e.preventDefault();
  document.getElementById('extractDrop').style.borderColor = 'var(--blue)';
  document.getElementById('extractDrop').style.background  = 'var(--blue-l)';
}
function extractDragLeave() {
  document.getElementById('extractDrop').style.borderColor = 'var(--gray-200)';
  document.getElementById('extractDrop').style.background  = '';
}
function extractDrop_(e) {
  e.preventDefault();
  extractDragLeave();
  const f = e.dataTransfer.files[0];
  if (f) setExtractFile(f);
}
function onExtractFileChange() {
  const f = document.getElementById('extractFile').files[0];
  if (f) setExtractFile(f);
}
function setExtractFile(file) {
  document.getElementById('extractDropPh').style.display  = 'none';
  document.getElementById('extractFileInfo').style.display = 'block';
  document.getElementById('extractFileName').textContent  = file.name;
  document.getElementById('extractFileSize').textContent  = (file.size / 1024).toFixed(1) + ' KB';
  document.getElementById('extractBtn').disabled = false;
  document.getElementById('extractResults').style.display = 'none';
  extractedSignatures = [];
}
function clearExtractFile(e) {
  e.stopPropagation();
  document.getElementById('extractFile').value = '';
  document.getElementById('extractDropPh').style.display  = 'block';
  document.getElementById('extractFileInfo').style.display = 'none';
  document.getElementById('extractBtn').disabled = true;
  document.getElementById('extractResults').style.display = 'none';
  extractedSignatures = [];
}

async function runExtract() {
  const fileInput = document.getElementById('extractFile');
  if (!fileInput.files[0]) return;

  const btn   = document.getElementById('extractBtn');
  const spin  = document.getElementById('extractSpinner');
  const label = document.getElementById('extractBtnLabel');

  btn.disabled = true;
  spin.style.display = 'inline-block';
  label.textContent  = 'Đang xử lý…';

  const form = new FormData();
  form.append('file', fileInput.files[0]);

  try {
    const res  = await fetch(`${API}/extract-signatures`, { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);

    extractedSignatures = data.signatures || [];
    renderExtractResults(data);
    showToast('success', 'Trích xuất thành công', `Tìm thấy ${data.total} chữ ký`);
  } catch (err) {
    showToast('error', 'Lỗi trích xuất', err.message);
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
    label.textContent  = '🔍 Trích xuất chữ ký';
  }
}

function renderExtractResults(data) {
  const results = document.getElementById('extractResults');
  const grid    = document.getElementById('extractGrid');
  const count   = document.getElementById('extractCount');

  count.textContent = `${data.total} chữ ký`;
  grid.innerHTML    = '';

  data.signatures.forEach((sig, idx) => {
    const card = document.createElement('div');
    card.style.cssText = 'background:var(--gray-50);border:1.5px solid var(--gray-200);border-radius:12px;overflow:hidden;';

    const header = document.createElement('div');
    header.style.cssText = 'padding:8px 12px;background:var(--gray-100);font-size:.78rem;font-weight:600;color:var(--gray-700);display:flex;align-items:center;justify-content:space-between';
    header.innerHTML = `<span>Trang ${sig.page}</span>
      <button onclick="downloadSig(${idx})" title="Tải về" style="background:none;border:none;cursor:pointer;font-size:.8rem;color:var(--blue);padding:2px 6px;border-radius:4px">⬇ Tải</button>`;

    const imgWrap = document.createElement('div');
    imgWrap.style.cssText = 'padding:12px;display:flex;justify-content:center;background:#fff';

    const img = document.createElement('img');
    img.src   = `data:${sig.mime};base64,${sig.image_b64}`;
    img.alt   = `Chữ ký trang ${sig.page}`;
    img.style.cssText = 'max-width:100%;max-height:160px;object-fit:contain;border-radius:6px;';

    imgWrap.appendChild(img);
    card.appendChild(header);
    card.appendChild(imgWrap);
    grid.appendChild(card);
  });

  results.style.display = 'block';
}

function downloadSig(idx) {
  const sig = extractedSignatures[idx];
  if (!sig) return;
  const a   = document.createElement('a');
  a.href    = `data:${sig.mime};base64,${sig.image_b64}`;
  a.download = `signature_page${sig.page}.png`;
  a.click();
}

function downloadAllSigs() {
  extractedSignatures.forEach((_, idx) => downloadSig(idx));
}
