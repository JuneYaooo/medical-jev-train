"""Generate the README figures (docs/figures/*.svg).

Numbers are copied from docs/EVALUATION.md (benchmark medical_decision_v1, 1,514 items,
mixed policy). Run: python scripts/make_readme_figures.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'figures'

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif"
MONO = "'SF Mono', Menlo, Consolas, 'PingFang SC', 'Noto Sans CJK SC', monospace"
INK, INK2, MUTED = '#1d1d1b', '#52514e', '#8a8984'
RULE, PANEL, SURFACE = '#d9d8d3', '#f6f5f2', '#ffffff'
BLUE, BLUE_LIGHT, BLUE_PALE = '#2a78d6', '#cde2fb', '#eaf2fd'
ORANGE, GRAY, GRAY_LIGHT = '#eb6834', '#6f6e69', '#c9c8c2'


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, fill=INK, weight=400, anchor='start', family=FONT, italic=False):
        style = ' font-style="italic"' if italic else ''
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
                 f'font-weight="{weight}" text-anchor="{anchor}"{style}>{escape(s)}</text>')

    def rect(self, x, y, w, h, fill=SURFACE, stroke=RULE, rx=6, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke=INK2, sw=1.5, arrow=False, dash=None):
        a = ' marker-end="url(#arrow)"' if arrow else ''
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{a}{d}/>')

    def path(self, d, stroke=INK2, sw=1.5, fill='none', arrow=False, dash=None):
        a = ' marker-end="url(#arrow)"' if arrow else ''
        ds = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{a}{ds}/>')

    def save(self, name, title):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(title)}">'
                f'<title>{escape(title)}</title>'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="{SURFACE}"/>')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(head + ''.join(self.parts) + '</svg>\n', encoding='utf-8')


def panel_title(s, x, y, letter, title):
    s.text(x, y, letter, size=20, weight=700)
    s.text(x + 22, y, title, size=16, weight=600)


def box(s, x, y, w, h, lines, fill=SURFACE, stroke=RULE, head_fill=INK, size=13, sw=1):
    """Box with a bold first line and lighter detail lines, centred."""
    s.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw)
    n = len(lines)
    lh = size + 6
    top = y + h / 2 - (n - 1) * lh / 2 + size * 0.36
    for i, t in enumerate(lines):
        s.text(x + w / 2, top + i * lh, t, size=size if i == 0 else size - 1,
               fill=head_fill if i == 0 else INK2, weight=600 if i == 0 else 400, anchor='middle')


# ---------------------------------------------------------------- figure 1: method
def method_figure():
    W, H = 1280, 640
    s = Svg(W, H)
    cols = [(24, 384), (432, 400), (856, 400)]  # x, width
    for x, w in cols:
        s.rect(x, 20, w, H - 40, fill=PANEL, stroke='none', rx=10)

    # (a) data
    x, w = cols[0]
    panel_title(s, x + 18, 54, 'a', '数据构建与隔离')
    bx, bw = x + 24, w - 48
    steps = [
        (78, 64, ['77 个公开题源', '医学考试题库 · 公开数据集 · 指南规则']),
        (172, 64, ['统一为选择题 JSON', 'state / instructions / criteria → 27 类决策任务']),
        (266, 82, ['三重隔离', '来源组 group_id · 精确输入哈希 · 归一化文本', '训练 ∩ 开发 ∩ 测试 ∩ 外部基准 = 0']),
    ]
    for y, h, lines in steps:
        box(s, bx, y, bw, h, lines)
    for y0, y1 in [(142, 172), (236, 266)]:
        s.line(x + w / 2, y0, x + w / 2, y1 - 2, arrow=True)
    # split into four sets
    s.line(x + w / 2, 348, x + w / 2, 376)
    s.line(bx + 40, 376, bx + bw - 40, 376)
    sets = [('训练池', '242,921'), ('开发集', '1,566'), ('留出测试', '1,557'), ('外部基准', '1,514')]
    sw_ = (bw - 3 * 8) / 4
    for i, (name, n) in enumerate(sets):
        sx = bx + i * (sw_ + 8)
        cx = sx + sw_ / 2
        s.line(cx, 376, cx, 392, arrow=True)
        fill, stroke = (BLUE_PALE, BLUE) if i == 0 else (SURFACE, RULE)
        box(s, sx, 394, sw_, 56, [name, n], fill=fill, stroke=stroke, size=13)
    s.text(bx + sw_ + 8 + (3 * sw_ + 16) / 2, 470, '↑ 仅评测，不参与训练', size=11, fill=MUTED, anchor='middle')
    # sampling
    s.line(bx + sw_ / 2, 450, bx + sw_ / 2, 498, arrow=True)
    box(s, bx, 500, bw, 64, ['按任务抽样', '每类 ≤3,000 题 + 通用医学 10,000 题'], fill=BLUE_PALE, stroke=BLUE)
    s.text(x + w / 2, 592, '数据全文公开：JuneYao/medical-jev-4b-data', size=11, fill=MUTED, anchor='middle')

    # (b) training + selection
    x, w = cols[1]
    panel_title(s, x + 18, 54, 'b', '训练与选模')
    bx, bw = x + 24, w - 48
    # model block: frozen base + LoRA
    s.rect(bx, 78, bw, 112, fill=SURFACE, stroke=RULE)
    s.text(bx + 16, 102, 'Qwen3.5-4B（冻结）', size=14, weight=600)
    s.text(bx + bw - 16, 102, '32 层', size=12, fill=MUTED, anchor='end')
    for i in range(8):
        lx = bx + 16 + i * ((bw - 32) / 8)
        lw = (bw - 32) / 8 - 6
        s.rect(lx, 114, lw, 22, fill=PANEL, stroke=RULE, rx=3)
        s.rect(lx + lw * 0.25, 140, lw * 0.5, 14, fill=BLUE, stroke='none', rx=3)
    s.text(bx + 16, 176, 'LoRA：全 32 层 · rank 16 · alpha 32 · 3,246 万参数', size=12, fill=INK2)
    s.line(x + w / 2, 190, x + w / 2, 212, arrow=True)
    box(s, bx, 214, bw, 64, ['选项打分损失', '答案字母 logits 交叉熵 + label smoothing 0.05', ], size=13)
    s.text(x + w / 2, 296, '每步随机循环打乱选项顺序 · FP16 · 2 epoch · lr 1e-4 cosine', size=11, fill=MUTED, anchor='middle')
    s.line(x + w / 2, 304, x + w / 2, 324, arrow=True)
    # selection loop
    s.rect(bx, 326, bw, 160, fill=SURFACE, stroke=BLUE, dash=None)
    s.text(bx + 16, 350, '选模（每 500 步）', size=14, weight=600, fill=BLUE)
    rows = [('1', '在开发集上评测（与训练隔离，不参与梯度）'),
            ('2', '保存开发集准确率最高的检查点'),
            ('3', '13 个检查点中选出第 5,500 步'),
            ('4', '导出约 130 MB 适配器 + 推理包')]
    for i, (k, t) in enumerate(rows):
        yy = 378 + i * 28
        s.add(f'<circle cx="{bx + 26}" cy="{yy - 4}" r="9" fill="{BLUE_PALE}" stroke="{BLUE}"/>')
        s.text(bx + 26, yy, k, size=11, weight=600, fill=BLUE, anchor='middle')
        s.text(bx + 44, yy, t, size=12.5, fill=INK)
    s.path(f'M{bx + bw - 10},460 C{bx + bw + 14},440 {bx + bw + 14},380 {bx + bw - 6},356', stroke=BLUE, sw=1.2, arrow=True)
    s.line(x + w / 2, 486, x + w / 2, 506, arrow=True)
    box(s, bx, 508, bw, 64, ['发布前验证', '对原始 4B 配对 bootstrap · 留出测试只评一次'], fill=PANEL, stroke=INK2)
    s.text(x + w / 2, 592, '44,847 题 · 2 epoch · 5,606 步 · RTX 5090 约 1.5 小时', size=11, fill=MUTED, anchor='middle')

    # (c) inference
    x, w = cols[2]
    panel_title(s, x + 18, 54, 'c', '推理：一次前向打分')
    bx, bw = x + 24, w - 48
    s.rect(bx, 78, bw, 132, fill=SURFACE, stroke=RULE)
    s.text(bx + 14, 100, '输入（JSON → 聊天模板）', size=12, fill=MUTED)
    lines = [('state', '发热、咳嗽；甲流核酸阳性，新冠阴性'),
             ('instructions', '按检测结果分类'),
             ('criteria', 'A 甲型流感   B 新冠感染   C 证据不足')]
    for i, (k, v) in enumerate(lines):
        yy = 126 + i * 26
        s.text(bx + 14, yy, k, size=12, fill=BLUE, family=MONO, weight=600)
        s.text(bx + 112, yy, v, size=12.5, fill=INK)
    s.line(x + w / 2, 210, x + w / 2, 232, arrow=True)
    box(s, bx, 234, bw, 52, ['Qwen3.5-4B + LoRA', '1 次前向，不生成文本'], fill=BLUE_PALE, stroke=BLUE)
    s.line(x + w / 2, 286, x + w / 2, 308, arrow=True)
    # probabilities bars
    s.rect(bx, 310, bw, 128, fill=SURFACE, stroke=RULE)
    s.text(bx + 14, 332, '下一个 token 的字母 logits → softmax', size=12, fill=MUTED)
    probs = [('A', '甲型流感', 0.9658), ('B', '新冠感染', 0.0173), ('C', '证据不足', 0.0169)]
    track_x, track_w = bx + 104, bw - 172
    for i, (k, lab, p) in enumerate(probs):
        yy = 352 + i * 28
        s.text(bx + 14, yy + 13, k, size=13, weight=700, family=MONO)
        s.text(bx + 32, yy + 13, lab, size=12.5, fill=INK2)
        s.rect(track_x, yy + 2, track_w, 14, fill=PANEL, stroke='none', rx=3)
        s.rect(track_x, yy + 2, max(track_w * p, 4), 14, fill=BLUE if i == 0 else GRAY_LIGHT, stroke='none', rx=3)
        s.text(track_x + track_w + 8, yy + 14, f'{p:.4f}', size=12, fill=INK, family=MONO)
    s.line(x + w / 2, 438, x + w / 2, 460, arrow=True)
    s.rect(bx, 462, bw, 52, fill=SURFACE, stroke=RULE)
    s.text(bx + 14, 484, '输出', size=12, fill=MUTED)
    s.text(bx + 14, 504, '{"answer":"influenza_a","probabilities":{…}}', size=11.5, fill=INK, family=MONO)
    s.rect(bx, 526, bw, 50, fill=PANEL, stroke=RULE, dash='4 3')
    s.text(bx + 14, 547, '候选 >52 时：分块淘汰', size=12.5, weight=600)
    s.text(bx + 14, 566, '每块 26 个、保留前 2，3 种分块取平均', size=12, fill=INK2)
    s.text(x + w / 2, 598, '无解析失败、不会编造选项；不输出解释', size=11, fill=MUTED, anchor='middle')

    s.save('method.svg', 'medical-jev-4b 方法总览：数据构建、训练选模与一次前向打分推理')


# ---------------------------------------------------------------- figure 2: results
OVERALL = [  # model, correct /1514, kind
    ('JEV 1.13（线上存档）', 1092, 'ref'),
    ('DeepSeek V4.1 Flash', 1076, 'ref'),
    ('medical-jev-4b v1', 1050, 'ours'),
    ('DeepSeek V4 Pro', 1041, 'ref'),
    ('Qwen3.8-27B', 1008, 'ref'),
    ('原始 Qwen3.5-4B', 925, 'base'),
    ('Qwen3.5-9B', 826, 'ref'),
]
TASKS = [  # name, base 4B, v1, DeepSeek Flash  (tasks with 100 items)
    ('诊断选择', 82, 89, 91), ('并发症', 77, 86, 89), ('检查选择', 81, 86, 81),
    ('病因', 80, 85, 84), ('用药选择', 67, 82, 80), ('治疗选择', 67, 77, 82),
    ('中医病机', 64, 77, 76), ('临床试验转介', 55, 62, 60), ('药物基因组建议', 39, 60, 54),
    ('模拟病例诊断', 55, 60, 77), ('疾病分期', 42, 54, 50), ('分诊', 55, 40, 61),
    ('中医证型（148 候选）', 18, 24, 37),
]


def axis_x(s, x0, x1, y_top, y_bot, vmin, vmax, ticks, fmt='{}'):
    for t in ticks:
        tx = x0 + (t - vmin) / (vmax - vmin) * (x1 - x0)
        s.line(tx, y_top, tx, y_bot, stroke='#ecebe7', sw=1)
        s.text(tx, y_bot + 18, fmt.format(t), size=11, fill=MUTED, anchor='middle')


def results_figure():
    W, H = 1280, 640
    s = Svg(W, H)

    # (a) overall accuracy
    panel_title(s, 24, 44, 'a', '外部基准总体准确率')
    s.text(24, 66, 'medical_decision_v1，1,514 题、27 类任务；失败与不支持均计错', size=12, fill=MUTED)
    lx, x0, x1 = 24, 212, 560
    top, bh, gap = 96, 30, 22
    vmax = 80
    yb = top + len(OVERALL) * (bh + gap) - gap
    axis_x(s, x0, x1, top - 8, yb + 6, 0, vmax, [0, 20, 40, 60, 80], '{}%')
    for i, (name, c, kind) in enumerate(OVERALL):
        y = top + i * (bh + gap)
        acc = c / 1514 * 100
        bw = (x1 - x0) * acc / vmax
        fill = BLUE if kind == 'ours' else (GRAY if kind == 'base' else GRAY_LIGHT)
        s.add(f'<path d="M{x0},{y} h{bw - 4} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h{-(bw - 4)} z" fill="{fill}"/>')
        s.text(x0 - 10, y + bh / 2 + 5, name, size=13, anchor='end',
               weight=700 if kind == 'ours' else 400, fill=INK)
        s.text(x0 + bw + 8, y + bh / 2 + 5, f'{acc:.2f}%', size=12.5, weight=700 if kind == 'ours' else 400,
               fill=INK, family=MONO)
    s.line(x0, top - 8, x0, yb + 6, stroke=INK2, sw=1)
    # annotation: v1 vs base
    yv = top + 2 * (bh + gap) + bh / 2
    yb4 = top + 5 * (bh + gap) + bh / 2
    ax = x1 + 34
    s.path(f'M{ax - 8},{yv} h8 V{yb4} h-8', stroke=BLUE, sw=1.4)
    s.text(ax + 8, (yv + yb4) / 2 - 6, '+8.26 pp', size=14, weight=700, fill=BLUE)
    s.text(ax + 8, (yv + yb4) / 2 + 13, '95% CI [5.74, 10.88]', size=11, fill=INK2)
    s.text(ax + 8, (yv + yb4) / 2 + 29, '同一打分方式', size=11, fill=INK2)
    s.text(24, H - 46, '参考模型通过 API 生成作答（关闭思考、temperature 0），本模型为本地选项打分；', size=11, fill=MUTED)
    s.text(24, H - 28, '作答方式与服务条件不同，对比仅供参考。该基准开发中反复使用，不是盲测。', size=11, fill=MUTED)

    # (b) per-task dumbbell
    px = 700
    panel_title(s, px, 44, 'b', '逐任务准确率（每类 100 题）')
    # legend
    ly = 70
    s.add(f'<circle cx="{px + 6}" cy="{ly - 4}" r="5" fill="{SURFACE}" stroke="{GRAY}" stroke-width="2"/>')
    s.text(px + 18, ly, '原始 Qwen3.5-4B', size=12, fill=INK2)
    s.add(f'<circle cx="{px + 140}" cy="{ly - 4}" r="6" fill="{BLUE}"/>')
    s.text(px + 152, ly, 'medical-jev-4b v1', size=12, fill=INK2)
    s.add(f'<path d="M{px + 284},{ly - 10} l6,6 l-6,6 l-6,-6 z" fill="{ORANGE}"/>')
    s.text(px + 296, ly, 'DeepSeek V4.1 Flash', size=12, fill=INK2)
    tx0, tx1 = px + 150, 1196
    s.text(1250, 70, 'v1', size=12, weight=700, fill=BLUE, anchor='end')
    top, row = 98, 34
    yb = top + len(TASKS) * row - row / 2
    axis_x(s, tx0, tx1, top - 12, yb + 4, 0, 100, [0, 25, 50, 75, 100], '{}%')
    X = lambda v: tx0 + v / 100 * (tx1 - tx0)
    for i, (name, b, v, d) in enumerate(TASKS):
        y = top + i * row
        weak = name in ('分诊', '模拟病例诊断', '中医证型（148 候选）', '疾病分期')
        if weak:
            s.rect(px - 4, y - row / 2 + 2, 1256 - px + 4, row - 4, fill='#fbf3ef', stroke='none', rx=4)
        s.text(tx0 - 12, y + 4.5, name, size=12.5, anchor='end', fill=INK)
        s.line(X(b), y, X(v), y, stroke=BLUE if v >= b else '#e34948', sw=2.5)
        s.add(f'<path d="M{X(d)},{y - 7} l7,7 l-7,7 l-7,-7 z" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="2"/>')
        s.add(f'<circle cx="{X(b)}" cy="{y}" r="5" fill="{SURFACE}" stroke="{GRAY}" stroke-width="2"/>')
        s.add(f'<circle cx="{X(v)}" cy="{y}" r="6.5" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
        s.text(1250, y + 4.5, str(v), size=12, fill=BLUE, weight=700, anchor='end', family=MONO)
    s.line(tx0, top - 12, tx0, yb + 4, stroke=INK2, sw=1)
    s.rect(px, H - 50, 14, 12, fill='#fbf3ef', stroke='none', rx=2)
    s.text(px + 20, H - 40, '已知短板（见正文）；分诊不应使用本模型', size=11, fill=MUTED)
    s.line(px + 250, H - 44, px + 272, H - 44, stroke='#e34948', sw=2.5)
    s.text(px + 278, H - 40, 'v1 低于原始 4B', size=11, fill=MUTED)
    s.text(px, H - 22, '其余 14 类任务题数 <100，见 docs/EVALUATION.md', size=11, fill=MUTED)

    s.save('results.svg', 'medical-jev-4b 评测结果：外部基准总体准确率与逐任务对比')


if __name__ == '__main__':
    method_figure()
    results_figure()
    print('wrote', *sorted(p.name for p in OUT.glob('*.svg')))
