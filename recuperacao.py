import os
import json
import argparse
import warnings
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import convolve

warnings.filterwarnings('ignore')

try:
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2hsv
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image', '-q'])
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2hsv


INTEL_CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

CLASS_COLORS = {
    'buildings': '#ff6b35',
    'forest':    '#44cc44',
    'glacier':   '#00cfff',
    'mountain':  '#aa88ff',
    'sea':       '#0077ff',
    'street':    '#ffcc00',
}


IMG_SIZE      = 150   
FEAT_IMG_SIZE = 64    
BG            = '#080818'


def find_intel_dirs(root: Path) -> tuple[Path, Path]:
   
    def _has_class_subfolders(p: Path) -> bool:
        return any((p / c).is_dir() for c in INTEL_CLASSES)

    train_dir = test_dir = None


    candidates_train = [
        root / 'seg_train' / 'seg_train',
        root / 'seg_train',
        root / 'archive' / 'seg_train' / 'seg_train',
        root / 'archive' / 'seg_train',
    ]
    candidates_test = [
        root / 'seg_test' / 'seg_test',
        root / 'seg_test',
        root / 'archive' / 'seg_test' / 'seg_test',
        root / 'archive' / 'seg_test',
    ]

    for p in candidates_train:
        if p.exists() and _has_class_subfolders(p):
            train_dir = p
            break

    for p in candidates_test:
        if p.exists() and _has_class_subfolders(p):
            test_dir = p
            break

    
    if train_dir is None or test_dir is None:
        for p in sorted(root.rglob('*')):
            if p.is_dir() and _has_class_subfolders(p):
                name = p.name.lower()
                if train_dir is None and 'train' in name:
                    train_dir = p
                if test_dir is None and 'test' in name:
                    test_dir = p

    if train_dir is None:
        raise FileNotFoundError(
            f"Não encontrei seg_train dentro de '{root}'.\n"
            f"Certifique-se de extrair o zip do Kaggle:\n"
            f"  unzip intel-image-classification.zip -d ./intel\n"
            f"E passar a pasta raiz com --data ./intel"
        )
    if test_dir is None:
        print("[aviso] seg_test não encontrado — usando seg_train como queries também.")
        test_dir = train_dir

    return train_dir, test_dir


def load_intel_dataset(train_dir: Path, test_dir: Path,
                       n_docs: int, n_queries: int,
                       seed: int = 42) -> dict:
  
    rng = random.Random(seed)

    def _center_bbox(w, h):
        m = 0.15
        return [int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))]

    def _sample_images(base_dir: Path, n: int) -> list[Path]:
        imgs = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            imgs.extend(base_dir.glob(ext))
        rng.shuffle(imgs)
        return imgs[:n]

    documents = {}
    queries   = []

    classes_found = [c for c in INTEL_CLASSES if (train_dir / c).is_dir()]
    if not classes_found:
        raise FileNotFoundError(
            f"Nenhuma subpasta de classe encontrada em '{train_dir}'.\n"
            f"Esperado: {INTEL_CLASSES}"
        )

    print(f"[dataset] Classes encontradas: {classes_found}")

    for cat in classes_found:
        # Documentos (seg_train)
        doc_imgs = _sample_images(train_dir / cat, n_docs)
        for p in doc_imgs:
            try:
                w, h = Image.open(p).size
            except Exception:
                w, h = IMG_SIZE, IMG_SIZE
            documents[p.name] = {
                'category': cat,
                'path':     str(p),
                'bbox':     _center_bbox(w, h),
            }

        # Queries (seg_test)
        test_class_dir = test_dir / cat
        if test_class_dir.is_dir():
            q_imgs = _sample_images(test_class_dir, n_queries)
            for p in q_imgs:
                try:
                    w, h = Image.open(p).size
                except Exception:
                    w, h = IMG_SIZE, IMG_SIZE
                queries.append({
                    'filename': p.name,
                    'category': cat,
                    'path':     str(p),
                    'bbox':     _center_bbox(w, h),
                })

    print(f"[dataset] {len(documents)} documentos | {len(queries)} queries")
    return {'documents': documents, 'queries': queries}



def extract_features(pil_img: Image.Image) -> np.ndarray:
    
    img = pil_img.resize((FEAT_IMG_SIZE, FEAT_IMG_SIZE)).convert('RGB')
    arr = np.array(img)
    gray = np.array(img.convert('L')) / 255.0


    feat_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    lbp      = local_binary_pattern(gray, P=8, R=1.5, method='uniform')
    feat_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    hsv = rgb2hsv(arr)
    feat_hsv_hist = np.concatenate([
        np.histogram(hsv[:, :, c], bins=32, range=(0, 1), density=True)[0]
        for c in range(3)
    ])

    feat_moments = []
    for c in range(3):
        ch = hsv[:, :, c].ravel()
        mu = ch.mean(); sigma = ch.std() + 1e-9
        skew = float(np.mean(((ch - mu) / sigma) ** 3))
        feat_moments.extend([mu, sigma, skew])
    feat_moments = np.array(feat_moments)

    return np.concatenate([feat_hog, feat_lbp, feat_hsv_hist, feat_moments])



def generate_proposals(H: int, W: int) -> list:
  
    proposals = set()

    proposals.add((0, 0, W, H))

    # Bandas horizontais (estrutura de cena: céu / meio / chão)
    for band in range(3):
        y1 = band * H // 3
        y2 = (band + 1) * H // 3
        proposals.add((0, y1, W, y2))

    # Janelas em múltiplas escalas
    for scale in [0.5, 0.65, 0.8]:
        rh, rw = int(H * scale), int(W * scale)
        for fy in np.linspace(0, 1, 4):
            for fx in np.linspace(0, 1, 4):
                y = int(fy * (H - rh))
                x = int(fx * (W - rw))
                proposals.add((
                    max(0, x), max(0, y),
                    min(W, x + rw), min(H, y + rh)
                ))

    return list(proposals)


def compute_iou(b1: list, b2: list) -> float:
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0



def build_index(documents: dict) -> tuple:
  
    print("[indexação] Extraindo descritores...")
    index = []
    n_docs = len(documents)

    for i, (fname, info) in enumerate(documents.items(), 1):
        if i % 25 == 0 or i == n_docs:
            print(f"  {i}/{n_docs} imagens processadas...")

        try:
            img = Image.open(info['path']).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"  [aviso] Não foi possível abrir '{info['path']}': {e}")
            continue

        gt_bbox  = info['bbox']
        proposals = generate_proposals(IMG_SIZE, IMG_SIZE)

        for bbox in proposals:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            feat = extract_features(crop)
            index.append({
                'filename':    fname,
                'category':    info['category'],
                'path':        info['path'],
                'gt_bbox':     gt_bbox,
                'region_bbox': list(bbox),
                'feat':        feat,
            })

    if not index:
        raise RuntimeError("Nenhuma região indexada. Verifique os caminhos das imagens.")

    feats_matrix = np.vstack([e['feat'] for e in index])
    feats_norm   = normalize(feats_matrix)

    print(f"[indexação] {len(index)} regiões | {len(documents)} imagens | dim={feats_matrix.shape[1]}")
    return index, feats_norm


def search(query_img: Image.Image,
           query_bbox: list,
           index: list,
           feats_norm: np.ndarray,
           alpha: float = 0.9,
           top_k: int = 10) -> list:
 
    beta = 1.0 - alpha

    q_feat = extract_features(query_img.resize((IMG_SIZE, IMG_SIZE)))
    q_norm = normalize(q_feat.reshape(1, -1))

    vis_sim = cosine_similarity(q_norm, feats_norm)[0]
    sp_iou  = np.array([compute_iou(query_bbox, e['region_bbox']) for e in index])
    scores  = alpha * vis_sim + beta * sp_iou

    # Colapsa para imagens únicas mantendo melhor score por imagem
    seen = {}
    for i in np.argsort(-scores):
        fn = index[i]['filename']
        if fn not in seen:
            seen[fn] = {
                'filename':    fn,
                'category':    index[i]['category'],
                'path':        index[i]['path'],
                'vis_sim':     float(vis_sim[i]),
                'iou':         float(sp_iou[i]),
                'combined':    float(scores[i]),
                'region_bbox': index[i]['region_bbox'],
                'gt_bbox':     index[i]['gt_bbox'],
            }
        if len(seen) >= top_k * 4:
            break

    return list(seen.values())



def compute_metrics(ranked: list, query_cat: str,
                    n_relevant: int, ks=(5, 10)) -> dict:
    labels = [1 if r['category'] == query_cat else 0 for r in ranked]

    def pat(k): return sum(labels[:k]) / k if k and k <= len(labels) else 0
    def rat(k): return sum(labels[:k]) / n_relevant if n_relevant else 0

    metrics = {}
    for k in ks:
        p = pat(k); r = rat(k)
        metrics[f'precision@{k}'] = p
        metrics[f'recall@{k}']    = r
        metrics[f'F1@{k}']        = 2 * p * r / (p + r + 1e-9)

    ap, hits = 0.0, 0
    for k, lb in enumerate(labels, 1):
        if lb:
            hits += 1
            ap   += hits / k
    metrics['AP']           = ap / n_relevant if n_relevant else 0
    metrics['PR_precision'] = [pat(k) for k in range(1, len(labels) + 1)]
    metrics['PR_recall']    = [rat(k) for k in range(1, len(labels) + 1)]
    return metrics


def alpha_sweep(queries: list, documents: dict,
                index: list, feats_norm: np.ndarray) -> dict:
    """Testa α ∈ {0.0, 0.1, …, 1.0} e retorna o que maximiza mAP."""
    print("\n[sweep] Buscando melhor α...")
    best_a, best_map = 0.9, 0.0
    sweep = {}

    for alpha in np.round(np.arange(0.0, 1.05, 0.1), 1):
        aps = []
        for q in queries:
            try:
                qimg = Image.open(q['path']).convert('RGB')
            except Exception:
                continue
            ranked = search(qimg, q['bbox'], index, feats_norm,
                            alpha=float(alpha), top_k=25)
            n_rel  = sum(1 for d in documents.values()
                         if d['category'] == q['category'])
            m      = compute_metrics(ranked, q['category'], n_rel)
            aps.append(m['AP'])

        mAP = float(np.mean(aps)) if aps else 0.0
        sweep[float(alpha)] = round(mAP, 4)
        marker = ' ◄ BEST' if mAP > best_map else ''
        if mAP > best_map:
            best_map = mAP; best_a = float(alpha)
        print(f"   α={alpha:.1f}  mAP={mAP:.4f}{marker}")

    print(f"\n[sweep] Melhor: α={best_a:.1f}  mAP={best_map:.4f}")
    return {'sweep': sweep, 'best_alpha': best_a, 'best_mAP': best_map}


def _style_ax(ax):
    ax.set_facecolor('#0d0d22')
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334')


def _load_img(path: str, size: int = IMG_SIZE) -> Image.Image | None:
    try:
        return Image.open(path).convert('RGB').resize((size, size))
    except Exception:
        return None


def plot_dashboard(all_results: dict, sweep_data: dict,
                   sim_matrix: np.ndarray, img_cats: list,
                   out_path: Path):
    """Dashboard PNG com métricas, curvas e strips de recuperação."""
    cats = list(all_results.keys())
    mAP  = np.mean([all_results[c]['AP'] for c in cats])

    fig  = plt.figure(figsize=(26, 20), facecolor=BG)
    fig.suptitle(
        'CBIR — Intel Image Classification  |  Avaliação Completa',
        fontsize=18, fontweight='bold', color='white', y=0.99
    )

    outer = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.40, wspace=0.30,
        top=0.96, bottom=0.04,
        left=0.04, right=0.97,
    )

    # ── A: AP por classe ──────────────────────────────────────
    ax1 = fig.add_subplot(outer[0, 0]); _style_ax(ax1)
    ap_vals = [all_results[c]['AP'] for c in cats]
    bars = ax1.bar(cats, ap_vals,
                   color=[CLASS_COLORS[c] for c in cats],
                   width=0.55, edgecolor='white', linewidth=0.5)
    ax1.axhline(mAP, color='white', ls='--', lw=1.5, alpha=0.7)
    ax1.text(len(cats) - 0.5, mAP + 0.03,
             f'mAP={mAP:.3f}', color='white', fontsize=8, ha='right')
    for b, v in zip(bars, ap_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}',
                 ha='center', color='white', fontsize=8, fontweight='bold')
    ax1.set_ylim(0, 1.25)
    ax1.set_title('Average Precision por Classe', color='#aaaaff', fontsize=9, pad=4)
    ax1.set_xticklabels(cats, rotation=25, fontsize=7, color='white')

    # ── B: Sweep α ───────────────────────────────────────────
    ax2 = fig.add_subplot(outer[0, 1]); _style_ax(ax2)
    alphas = [float(a) for a in sweep_data['sweep']]
    maps   = list(sweep_data['sweep'].values())
    ax2.plot(alphas, maps, 'o-', color='#44ccff', lw=2, ms=5)
    ax2.axvline(sweep_data['best_alpha'], color='#ffcc44', ls='--', lw=1.5)
    ax2.text(sweep_data['best_alpha'] + 0.02, min(maps) + 0.005,
             f"α*={sweep_data['best_alpha']:.1f}", color='#ffcc44', fontsize=8)
    ax2.fill_between(alphas, maps, alpha=0.15, color='#44ccff')
    ax2.set_xlabel('α (peso visual)', color='#aaa', fontsize=8)
    ax2.set_ylabel('mAP', color='#aaa', fontsize=8)
    ax2.set_title('Sweep α/β  (β = 1 − α)', color='#aaaaff', fontsize=9, pad=4)

    # ── C: P / R / F1 @5 e @10 ──────────────────────────────
    ax3 = fig.add_subplot(outer[0, 2]); _style_ax(ax3)
    x  = np.arange(len(cats)); w = 0.15
    metrics_to_plot = [
        ('precision@5',  '#4488ff', 'P@5'),
        ('recall@5',     '#ff8844', 'R@5'),
        ('F1@5',         '#44ee88', 'F1@5'),
        ('precision@10', '#4488ff', 'P@10'),
        ('recall@10',    '#ff8844', 'R@10'),
    ]
    offsets = np.linspace(-2 * w, 2 * w, len(metrics_to_plot))
    for off, (key, color, label) in zip(offsets, metrics_to_plot):
        vals = [all_results[c][key] for c in cats]
        alpha_bar = 0.9 if '@5' in label else 0.45
        ax3.bar(x + off, vals, w, color=color, alpha=alpha_bar, label=label)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cats, rotation=25, fontsize=7, color='white')
    ax3.set_ylim(0, 1.3)
    ax3.legend(fontsize=6.5, framealpha=0.2, labelcolor='white',
               loc='upper right', ncol=2)
    ax3.set_title('P / R / F1  @5 e @10', color='#aaaaff', fontsize=9, pad=4)

    # ── D: Matriz de similaridade ─────────────────────────────
    ax4 = fig.add_subplot(outer[0, 3]); ax4.set_facecolor('#0d0d22')
    if sim_matrix is not None and len(img_cats) > 0:
        cmap  = LinearSegmentedColormap.from_list(
            'intel', ['#080818', '#0077ff', '#44cc44', '#ffcc44'])
        order = np.argsort(img_cats)
        sim_s = sim_matrix[order][:, order]
        im    = ax4.imshow(sim_s, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(
            colors='white', labelsize=6)
        cats_ord = [img_cats[i] for i in order]
        for i in range(1, len(cats_ord)):
            if cats_ord[i] != cats_ord[i - 1]:
                ax4.axhline(i - 0.5, color='white', lw=0.8, alpha=0.4)
                ax4.axvline(i - 0.5, color='white', lw=0.8, alpha=0.4)
    ax4.set_title('Matriz Cosine Similarity\n(documentos, ordenados por classe)',
                  color='#aaaaff', fontsize=9, pad=4)
    ax4.set_xticks([]); ax4.set_yticks([])

    # ── E: Curvas P-R ────────────────────────────────────────
    ax5 = fig.add_subplot(outer[1, 0:2]); _style_ax(ax5)
    for cat in cats:
        r = all_results[cat]
        ax5.plot(r['PR_recall'], r['PR_precision'], '-o',
                 color=CLASS_COLORS[cat], lw=2, ms=3,
                 label=f'{cat}  (AP={r["AP"]:.3f})')
    ax5.set_xlabel('Recall', color='#aaa', fontsize=9)
    ax5.set_ylabel('Precision', color='#aaa', fontsize=9)
    ax5.set_xlim(-0.02, 1.02); ax5.set_ylim(-0.02, 1.15)
    ax5.legend(fontsize=8.5, framealpha=0.25, labelcolor='white', loc='upper right')
    ax5.set_title('Curvas Precision–Recall', color='#aaaaff', fontsize=10, pad=4)
    ax5.grid(True, color='#223', alpha=0.4)

    # ── F: Precision@K ───────────────────────────────────────
    ax6 = fig.add_subplot(outer[1, 2:4]); _style_ax(ax6)
    for cat in cats:
        lbs = [1 if r['category'] == cat else 0
               for r in all_results[cat]['ranked']]
        ks  = list(range(1, len(lbs) + 1))
        pk  = [sum(lbs[:k]) / k for k in ks]
        ax6.plot(ks, pk, color=CLASS_COLORS[cat], lw=2, label=cat)
    ax6.set_xlabel('K', color='#aaa', fontsize=9)
    ax6.set_ylabel('Precision@K', color='#aaa', fontsize=9)
    ax6.set_title('Precision@K', color='#aaaaff', fontsize=10, pad=4)
    ax6.legend(fontsize=8.5, framealpha=0.25, labelcolor='white', loc='upper right')
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, color='#223', alpha=0.4)

    # ── G: Strips de recuperação (1 por classe) ───────────────
    ret = gridspec.GridSpecFromSubplotSpec(
        len(cats), 6, subplot_spec=outer[2, :],
        hspace=0.06, wspace=0.04)

    for qi, cat in enumerate(cats):
        r     = all_results[cat]
        color = CLASS_COLORS[cat]

        # Query
        axq = fig.add_subplot(ret[qi, 0])
        qimg = _load_img(r['_query_path'])
        if qimg:
            axq.imshow(qimg)
        else:
            axq.set_facecolor('#222')
        qb = r['query_bbox']
        axq.add_patch(mpatches.Rectangle(
            (qb[0], qb[1]), qb[2] - qb[0], qb[3] - qb[1],
            lw=2, edgecolor=color, facecolor='none'))
        axq.set_title(f'QUERY\n{cat}', color=color,
                      fontsize=6.5, pad=1, fontweight='bold')
        axq.axis('off')
        for sp in axq.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(2)

        # Top-5
        for ri, res in enumerate(r['ranked'][:5]):
            ax = fig.add_subplot(ret[qi, ri + 1])
            img2 = _load_img(res['path'])
            if img2:
                ax.imshow(img2)
            else:
                ax.set_facecolor('#222')
            rb = res['region_bbox']
            ec = '#44ff88' if res['category'] == cat else '#ff4444'
            ax.add_patch(mpatches.Rectangle(
                (rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
                lw=1.2, edgecolor=ec, facecolor='none', ls='--'))
            mark = '✓' if res['category'] == cat else '✗'
            ax.set_title(
                f'{mark} {res["category"][:5]}\n{res["combined"]:.3f}',
                color=ec, fontsize=6, pad=1)
            ax.axis('off')

    fig.text(
        0.01, 0.005,
        f'Intel Image Classification  |  HOG + LBP + HSV  |  '
        f'Sliding Window multi-escala  |  '
        f'Score = α·CosSim + β·IoU  |  '
        f'α={sweep_data["best_alpha"]:.1f}  |  mAP={mAP:.3f}',
        color='#445', fontsize=7,
    )

    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor=BG)
    print(f"[saída] Dashboard: {out_path}")
    plt.close()


def generate_pdf(all_results: dict, sweep_data: dict,
                 sim_matrix, img_cats, out_path: Path):
    """Relatório PDF completo."""
    cats = list(all_results.keys())
    mAP  = np.mean([all_results[c]['AP'] for c in cats])

    with PdfPages(out_path) as pdf:

        # ── Capa + tabela de métricas ─────────────────────────
        fig = plt.figure(figsize=(11.7, 8.27), facecolor=BG)
        fig.text(0.5, 0.93,
                 'CBIR — Intel Image Classification',
                 ha='center', fontsize=22, fontweight='bold', color='white')
        fig.text(0.5, 0.87,
                 'Content-Based Image Retrieval com Ponderação Espacial (IoU)',
                 ha='center', fontsize=13, color='#aaaaff')
        fig.text(0.5, 0.82,
                 '6 classes: buildings | forest | glacier | mountain | sea | street',
                 ha='center', fontsize=10, color='#888899')

        rows = [['Classe', 'P@5', 'P@10', 'R@5', 'R@10', 'F1@5', 'AP']]
        for c in cats:
            v = all_results[c]
            rows.append([c,
                         f"{v['precision@5']:.2f}",
                         f"{v['precision@10']:.2f}",
                         f"{v['recall@5']:.2f}",
                         f"{v['recall@10']:.2f}",
                         f"{v['F1@5']:.2f}",
                         f"{v['AP']:.3f}"])
        rows.append(['MEAN',
                     f"{np.mean([all_results[c]['precision@5'] for c in cats]):.2f}",
                     f"{np.mean([all_results[c]['precision@10'] for c in cats]):.2f}",
                     f"{np.mean([all_results[c]['recall@5'] for c in cats]):.2f}",
                     f"{np.mean([all_results[c]['recall@10'] for c in cats]):.2f}",
                     f"{np.mean([all_results[c]['F1@5'] for c in cats]):.2f}",
                     f"{mAP:.3f}"])

        ax_t = fig.add_axes([0.05, 0.05, 0.90, 0.65])
        ax_t.axis('off')
        tbl = ax_t.table(cellText=rows[1:], colLabels=rows[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor('#334')
            if ri == 0:
                cell.set_facecolor('#1a1a4e')
                cell.set_text_props(color='white', fontweight='bold')
            elif ri == len(rows) - 1:
                cell.set_facecolor('#2a1a0e')
                cell.set_text_props(color='#ffcc44', fontweight='bold')
            else:
                cat = cats[ri - 1] if ri <= len(cats) else None
                cell.set_facecolor('#0f0f20')
                color = CLASS_COLORS.get(cat, 'white') if ci == 0 else 'white'
                cell.set_text_props(color=color)
        pdf.savefig(fig, facecolor=BG); plt.close()

        # ── Gráficos de avaliação ─────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.27), facecolor=BG)
        fig.suptitle('Gráficos de Avaliação',
                     fontsize=14, fontweight='bold', color='white', y=0.99)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                               top=0.93, bottom=0.06, left=0.07, right=0.97)

        # AP
        ax = fig.add_subplot(gs[0, 0]); _style_ax(ax)
        ap_vals = [all_results[c]['AP'] for c in cats]
        bars = ax.bar(cats, ap_vals, color=[CLASS_COLORS[c] for c in cats],
                      width=0.55, edgecolor='white', lw=0.5)
        ax.axhline(mAP, color='white', ls='--', lw=1.5)
        for b, v in zip(bars, ap_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}',
                    ha='center', color='white', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_title(f'Average Precision  (mAP={mAP:.3f})',
                     color='#aaaaff', fontsize=9)
        ax.set_xticklabels(cats, rotation=25, fontsize=7, color='white')

        # Sweep
        ax = fig.add_subplot(gs[0, 1]); _style_ax(ax)
        alphas = [float(a) for a in sweep_data['sweep']]
        maps   = list(sweep_data['sweep'].values())
        ax.plot(alphas, maps, 'o-', color='#44ccff', lw=2, ms=5)
        ax.axvline(sweep_data['best_alpha'], color='#ffcc44', ls='--', lw=1.5)
        ax.fill_between(alphas, maps, alpha=0.15, color='#44ccff')
        ax.set_xlabel('α', color='#aaa', fontsize=8)
        ax.set_ylabel('mAP', color='#aaa', fontsize=8)
        ax.set_title('Sweep α/β', color='#aaaaff', fontsize=9)

        # P-R
        ax = fig.add_subplot(gs[1, 0]); _style_ax(ax)
        for cat in cats:
            r = all_results[cat]
            ax.plot(r['PR_recall'], r['PR_precision'], '-o',
                    color=CLASS_COLORS[cat], lw=2, ms=3,
                    label=f'{cat} ({r["AP"]:.3f})')
        ax.set_xlabel('Recall', color='#aaa', fontsize=8)
        ax.set_ylabel('Precision', color='#aaa', fontsize=8)
        ax.set_title('Curvas P-R', color='#aaaaff', fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor='white')
        ax.grid(True, color='#223', alpha=0.4)

        # P@K
        ax = fig.add_subplot(gs[1, 1]); _style_ax(ax)
        for cat in cats:
            lbs = [1 if r['category'] == cat else 0
                   for r in all_results[cat]['ranked']]
            ks  = list(range(1, len(lbs) + 1))
            pk  = [sum(lbs[:k]) / k for k in ks]
            ax.plot(ks, pk, color=CLASS_COLORS[cat], lw=2, label=cat)
        ax.set_xlabel('K', color='#aaa', fontsize=8)
        ax.set_ylabel('P@K', color='#aaa', fontsize=8)
        ax.set_title('Precision@K', color='#aaaaff', fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor='white')
        ax.set_ylim(0, 1.1); ax.grid(True, color='#223', alpha=0.4)

        pdf.savefig(fig, facecolor=BG); plt.close()

        # ── Uma página por query ──────────────────────────────
        for cat in cats:
            r     = all_results[cat]
            color = CLASS_COLORS[cat]

            fig = plt.figure(figsize=(11.7, 8.27), facecolor=BG)
            fig.suptitle(
                f'Query: {cat.upper()}  |  '
                f'AP={r["AP"]:.3f}  P@5={r["precision@5"]:.2f}  '
                f'R@5={r["recall@5"]:.2f}  F1@5={r["F1@5"]:.2f}',
                fontsize=11, fontweight='bold', color=color, y=0.99)

            gs2 = gridspec.GridSpec(2, 6, figure=fig,
                                    hspace=0.38, wspace=0.06,
                                    top=0.93, bottom=0.04,
                                    left=0.02, right=0.98)

            # Query
            axq = fig.add_subplot(gs2[0, 0])
            qimg = _load_img(r['_query_path'])
            if qimg:
                axq.imshow(qimg)
            else:
                axq.set_facecolor('#222')
            qb = r['query_bbox']
            axq.add_patch(mpatches.Rectangle(
                (qb[0], qb[1]), qb[2] - qb[0], qb[3] - qb[1],
                lw=2.5, edgecolor=color, facecolor='none'))
            axq.set_title('QUERY', color=color,
                          fontsize=8, fontweight='bold', pad=2)
            axq.axis('off')
            for sp in axq.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color); sp.set_linewidth(2.5)

            # Top-11
            for ri, res in enumerate(r['ranked'][:11]):
                row_i, col_i = (0, ri + 1) if ri < 5 else (1, ri - 5)
                ax = fig.add_subplot(gs2[row_i, col_i])
                img2 = _load_img(res['path'])
                if img2:
                    ax.imshow(img2)
                else:
                    ax.set_facecolor('#222')
                rb = res['region_bbox']
                ec = '#44ff88' if res['category'] == cat else '#ff4444'
                ax.add_patch(mpatches.Rectangle(
                    (rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
                    lw=1.5, edgecolor=ec, facecolor='none', ls='--'))
                gb = res['gt_bbox']
                ax.add_patch(mpatches.Rectangle(
                    (gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1],
                    lw=1, edgecolor='yellow', facecolor='none', alpha=0.3))
                mark = '✓' if res['category'] == cat else '✗'
                ax.set_title(
                    f'{mark} {res["category"]}\n'
                    f'V:{res["vis_sim"]:.2f}  I:{res["iou"]:.2f}\n'
                    f'{res["combined"]:.3f}',
                    color=ec, fontsize=6, pad=1)
                ax.axis('off')

            pdf.savefig(fig, facecolor=BG); plt.close()

    print(f"[saída] PDF: {out_path}")



def run(data_root: Path, n_docs: int, n_queries: int,
        top_k: int, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Localizar pastas do dataset Intel
    train_dir, test_dir = find_intel_dirs(data_root)
    print(f"[dataset] seg_train → {train_dir}")
    print(f"[dataset] seg_test  → {test_dir}")

    # 2. Carregar amostras
    meta      = load_intel_dataset(train_dir, test_dir, n_docs, n_queries)
    documents = meta['documents']
    queries   = meta['queries']

    # 3. Indexar
    index, feats_norm = build_index(documents)

    # 4. Sweep α/β
    sweep_data = alpha_sweep(queries, documents, index, feats_norm)
    best_alpha = sweep_data['best_alpha']

    # 5. Busca final
    print(f"\n[busca] Executando queries com α={best_alpha:.1f}...")
    all_results = {}

    # Agrupa queries por classe e pega a primeira de cada (1 query representativa)
    seen_cats: set = set()
    for q in queries:
        cat = q['category']
        if cat in seen_cats:
            continue
        seen_cats.add(cat)

        try:
            qimg = Image.open(q['path']).convert('RGB')
        except Exception as e:
            print(f"  [aviso] Não foi possível abrir query: {e}")
            continue

        ranked  = search(qimg, q['bbox'], index, feats_norm,
                         alpha=best_alpha, top_k=top_k * 4)
        n_rel   = sum(1 for d in documents.values() if d['category'] == cat)
        metrics = compute_metrics(ranked, cat, n_rel)

        all_results[cat] = {
            **metrics,
            'ranked':       ranked[:top_k],
            'query_bbox':   q['bbox'],
            '_query_path':  q['path'],
        }

        print(f"  [{cat}]  AP={metrics['AP']:.3f}  "
              f"P@5={metrics['precision@5']:.2f}  "
              f"R@5={metrics['recall@5']:.2f}  "
              f"F1@5={metrics['F1@5']:.2f}")

    if not all_results:
        print("[erro] Nenhuma query processada.")
        return

    mAP = np.mean([v['AP'] for v in all_results.values()])
    print(f"\n  mAP = {mAP:.4f}\n")

    # 6. Matriz de similaridade entre documentos
    print("[métricas] Calculando matriz de similaridade...")
    img_fnames = list(documents.keys())
    img_cats   = [documents[f]['category'] for f in img_fnames]
    img_feats  = []
    for fname in img_fnames:
        img2 = _load_img(documents[fname]['path'])
        feat = extract_features(img2) if img2 else np.zeros(index[0]['feat'].shape)
        img_feats.append(feat)
    img_FM     = normalize(np.vstack(img_feats))
    sim_matrix = cosine_similarity(img_FM)

    # 7. Salvar resultados JSON
    results_out = {}
    for cat, v in all_results.items():
        results_out[cat] = {k: val for k, val in v.items()
                            if k not in ('_query_path',)}
    (out_dir / 'results.json').write_text(json.dumps(results_out, indent=2))
    (out_dir / 'sweep.json').write_text(json.dumps(sweep_data, indent=2))
    np.save(out_dir / 'sim_matrix.npy', sim_matrix)

    # 8. Dashboard PNG
    plot_dashboard(all_results, sweep_data, sim_matrix, img_cats,
                   out_dir / 'cbir_intel_dashboard.png')

    # 9. Relatório PDF
    generate_pdf(all_results, sweep_data, sim_matrix, img_cats,
                 out_dir / 'cbir_intel_report.pdf')

    print(f"\n{'═'*62}")
    print(f"  CBIR — Intel Image Classification")
    print(f"  mAP = {mAP:.4f}")
    print(f"  Saídas em: '{out_dir}'")
    print(f"    cbir_intel_dashboard.png")
    print(f"    cbir_intel_report.pdf")
    print(f"    results.json  |  sweep.json  |  sim_matrix.npy")
    print(f"{'═'*62}")


def main():
    parser = argparse.ArgumentParser(
        description='CBIR para Intel Image Classification Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Passos para rodar:

  1. Baixe o dataset do Kaggle:
       pip install kaggle
       kaggle datasets download -d puneet6060/intel-image-classification
       unzip intel-image-classification.zip -d ./intel

  2. Execute o sistema:
       python cbir_intel.py --data ./intel

        """
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Pasta raiz do dataset Intel extraído (contém seg_train/ e seg_test/)'
    )
    parser.add_argument(
        '--docs', type=int, default=25,
        help='Documentos por classe a indexar (padrão: 25). '
             'Aumente para resultados melhores (ex: 100), diminua para testes rápidos (ex: 10).'
    )
    parser.add_argument(
        '--queries', type=int, default=5,
        help='Queries por classe do seg_test (padrão: 5). '
             'Apenas a 1ª de cada classe é usada como query representativa.'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Número de resultados retornados por query (padrão: 10)'
    )
    parser.add_argument(
        '--output', type=str, default='./saida',
        help='Pasta de saída (padrão: ./saida)'
    )

    args = parser.parse_args()

    run(
        data_root = Path(args.data),
        n_docs    = args.docs,
        n_queries = args.queries,
        top_k     = args.top_k,
        out_dir   = Path(args.output),
    )


if __name__ == '__main__':
    main()