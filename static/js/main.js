// ── Particles Background ─────────────────────────────────
(function initParticles() {
  const container = document.getElementById('particles');
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random()*100}%;
      animation-duration:${Math.random()*12+8}s;
      animation-delay:${Math.random()*8}s;
    `;
    container.appendChild(p);
  }
})();

// ── DOM ──────────────────────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImg     = document.getElementById('previewImg');
const btnRemove      = document.getElementById('btnRemove');
const btnPredict     = document.getElementById('btnPredict');
const loading        = document.getElementById('loading');
const resultCard     = document.getElementById('resultCard');

let selectedFile = null;

const CLASS_COLORS = {
  battery:    '#f97316', biological: '#22c55e', cardboard:  '#a16207',
  clothes:    '#a855f7', glass:      '#3b82f6', metal:      '#6b7280',
  paper:      '#eab308', plastic:    '#10b981', shoes:      '#b45309',
  trash:      '#ef4444'
};

const CLASS_EMOJI = {
  battery:    '🔋', biological: '🌿', cardboard:  '📦',
  clothes:    '👕', glass:      '🍶', metal:      '🔩',
  paper:      '📄', plastic:    '🧴', shoes:      '👟',
  trash:      '🗑️'
};

const CLASS_FA = {
  battery:    'fa-car-battery',
  biological: 'fa-leaf',
  cardboard:  'fa-box-open',
  clothes:    'fa-shirt',
  glass:      'fa-wine-bottle',
  metal:      'fa-gears',
  paper:      'fa-file-lines',
  plastic:    'fa-bottle-water',
  shoes:      'fa-shoe-prints',
  trash:      'fa-trash-can'
};

const CLASS_VI = {
  battery:    'Pin / Ắc quy',
  biological: 'Rác hữu cơ / Sinh học',
  cardboard:  'Giấy bìa / Carton',
  clothes:    'Quần áo / Vải',
  glass:      'Thủy tinh',
  metal:      'Kim loại',
  paper:      'Giấy',
  plastic:    'Nhựa',
  shoes:      'Giày dép',
  trash:      'Rác hỗn hợp'
};

// ── Events ───────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
btnRemove.addEventListener('click', e => { e.stopPropagation(); resetUI(); });
btnPredict.addEventListener('click', () => { if (selectedFile) uploadAndPredict(); });

const btnRetry = document.getElementById('btnRetry');
if (btnRetry) btnRetry.addEventListener('click', resetUI);

// ── Handle File ──────────────────────────────────────────
function handleFile(file) {
  const valid = ['image/jpeg','image/png','image/webp'];
  if (!valid.includes(file.type)) { alert('Only JPG, PNG, WEBP accepted!'); return; }
  if (file.size > 10*1024*1024) { alert('File too large! Max 10MB.'); return; }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropZone.style.display = 'none';
    previewSection.style.display = 'block';
    resultCard.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

// ── Upload & Predict ─────────────────────────────────────
async function uploadAndPredict() {
  btnPredict.disabled = true;
  loading.style.display = 'block';
  resultCard.style.display = 'none';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const bodyText = await res.text();
    let data = null;
    if (bodyText) {
      try { data = JSON.parse(bodyText); } catch (_) { data = null; }
    }

    if (!res.ok) {
      const msg = (data && data.error) ? data.error : (bodyText || `HTTP ${res.status}`);
      alert(msg);
      return;
    }

    if (!data) {
      alert('Server returned invalid response!');
      return;
    }

    if (data.error) { alert(data.error); return; }
    showResult(data);
  } catch (err) {
    alert('Server connection failed!');
    console.error(err);
  } finally {
    btnPredict.disabled = false;
    loading.style.display = 'none';
  }
}

// ── Show Result ──────────────────────────────────────────
function showResult(data) {
  const cls = data.predicted_class;
  const color = data.color || CLASS_COLORS[cls] || '#22c55e';

  // Badge
  const badge = document.getElementById('resultBadge');
  badge.style.background = color + '18';
  badge.style.border = `2px solid ${color}40`;
  const emoji = CLASS_EMOJI[cls] || '♻️';
  const emojiEl = document.getElementById('resultEmoji');
  const iconEl = document.getElementById('resultIcon');
  if (emojiEl) emojiEl.textContent = emoji;
  if (iconEl) {
    const fa = CLASS_FA[cls] || 'fa-recycle';
    iconEl.className = `fa-solid ${fa}`;
  }

  // Class name (VI primary, EN secondary)
  const labelEl = document.getElementById('resultLabel');
  const viName = CLASS_VI[cls] || data.label_vi || cls;
  labelEl.textContent = viName;
  labelEl.style.color = color;

  const viLabelEl = document.getElementById('resultVi');
  if (viLabelEl) {
    viLabelEl.textContent = '';
    viLabelEl.style.display = 'none';
  }

  // Meter
  const fill = document.getElementById('meterFill');
  fill.style.background = color;
  fill.style.color = color;
  requestAnimationFrame(() => { fill.style.width = data.confidence + '%'; });

  // Animated counter
  const valueEl = document.getElementById('meterValue');
  valueEl.style.color = color;
  animateCounter(valueEl, data.confidence);

  // Recycle badge
  const recycleBadge = document.getElementById('recycleBadge');
  if (data.recycle) {
    recycleBadge.className = 'info-row recyclable';
    recycleBadge.innerHTML = '<i class="fa-solid fa-recycle"></i> Có thể tái chế';
  } else {
    recycleBadge.className = 'info-row not-recyclable';
    recycleBadge.innerHTML = '<i class="fa-solid fa-xmark"></i> Không thể tái chế';
  }

  document.getElementById('resultTip').textContent = '💡 ' + data.tip;

  // Probability bars
  const chart = document.getElementById('probChart');
  chart.innerHTML = '';
  const sorted = Object.entries(data.all_probabilities).sort((a,b) => b[1]-a[1]);

  sorted.forEach(([name, prob], i) => {
    const c = CLASS_COLORS[name] || '#666';
    const displayName = CLASS_VI[name] || name;
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.style.animationDelay = `${0.5 + i * 0.05}s`;
    item.innerHTML = `
      <span class="prob-name">${displayName}</span>
      <div class="prob-track">
        <div class="prob-fill" style="background:${c}"></div>
      </div>
      <span class="prob-pct">${prob.toFixed(1)}%</span>
    `;
    chart.appendChild(item);

    requestAnimationFrame(() => {
      setTimeout(() => {
        item.querySelector('.prob-fill').style.width = Math.max(prob, 0.5) + '%';
      }, 600 + i * 50);
    });
  });

  // Hide preview, show result
  previewSection.style.display = 'none';
  resultCard.style.display = 'block';
}

// ── Counter Animation ────────────────────────────────────
function animateCounter(el, target) {
  let current = 0;
  const step = target / 40;
  const interval = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(interval); }
    el.textContent = current.toFixed(1) + '%';
  }, 20);
}

// ── Reset ────────────────────────────────────────────────
function resetUI() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewSection.style.display = 'none';
  resultCard.style.display = 'none';
  dropZone.style.display = 'block';

  // Reset meter
  const fill = document.getElementById('meterFill');
  fill.style.width = '0';
}
