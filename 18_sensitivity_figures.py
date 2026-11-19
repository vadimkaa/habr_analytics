"""Рисунки дополнительных проверок из сохранённых расчётных JSON."""
import json
import matplotlib
import matplotlib.pyplot as plt
from habrpub import BASE, RESULTS
from habrpub.sensitivity_public import snapshot, LABELS, CORPORA, HUBS, METRICS, FOLDS, POWER
matplotlib.use('Agg')

COLORS = ('#397bb9', '#d48a42')

def save(fig, name):
    fig.savefig(BASE / 'figs' / name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def error(ax, x, y, ci, color, **kw):
    ax.errorbar(x, y, xerr=[[x - ci[0]], [ci[1] - x]], fmt='o', ms=4.5, capsize=3, color=color, **kw)

def main():
    j = snapshot()
    cal = {c: j['corpora'][c]['calendar'] for c in CORPORA}
    old = {c: json.loads((RESULTS / f'part2_{c}_results.json').read_text()) for c in CORPORA}
    robust = json.loads((RESULTS / '09_robustness_results.json').read_text())['corpora']
    extra = json.loads((RESULTS / '14_extra_checks.json').read_text())['top_producers']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.16, 'axes.axisbelow': True})
    fig, ax = plt.subplots(figsize=(8, 3.1), layout='constrained')
    labels = []
    for i, c in enumerate(CORPORA):
        for k, threshold in enumerate(('3', '5')):
            q1, med, q3 = cal[c]['author_iqr'][threshold]
            y = 3 - 2 * i - k
            ax.plot([q1, q3], [y, y], lw=10, color=COLORS[i], alpha=0.35, solid_capstyle='butt')
            ax.plot(med, y, 'o', color=COLORS[i])
            labels.append(f'{LABELS[c]}, ≥{threshold} статьи')
    ax.set(yticks=[3, 2, 1, 0], yticklabels=labels, xlabel='VPD: 25-й, 50-й и 75-й перцентили авторских медиан')
    save(fig, '12_author_iqr.png')
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.6), gridspec_kw={'height_ratios': [3, 5]}, layout='constrained')
    upper = ['Все статьи', 'Одни и те же авторы*', 'Личные блоги']
    lower = ['Основная модель', 'Без плотности', 'Сплайн по возрасту', 'Подвыборка авторов**', 'С фикс. эффектами авторов**']
    for i, c in enumerate(CORPORA):
        a = old[c]['weekend']
        b = old[c]['paired_authors']
        d = extra[c]['personal_only']
        top = [(a['ratio'], a['ratio_ci_author']), (b['median_ratio'], b['ci95']), (d['weekend_ratio'], d['ci95_author'])]
        b = robust[c]['adjusted_weekend_no_density']
        bottom = [(cal[c]['ols']['ratio'], cal[c]['ols']['ci95']), (b['ratio_target_plus1'], b['ratio_ci95']), *[(cal[c][k]['ratio'], cal[c][k]['ci95']) for k in ('spline', 'same_sample_ols', 'fe')]]
        for ax, vals in zip(axs, (top, bottom)):
            for k, (v, ci) in enumerate(vals):
                error(ax, v, len(vals) - 1 - k + (0.12 if i == 0 else -0.12), ci, COLORS[i], label=LABELS[c] if k == 0 else None)
    for ax, names, title, lab in zip(axs, (upper, lower), ('А. Описательные отношения VPD', 'Б. Регрессии с поправками'), ('Отношение медиан VPD (*медиана внутриавторских отношений)', 'Отношение геометрических средних (просмотры + 1)')):
        ax.set(yticks=list(range(len(names) - 1, -1, -1)), yticklabels=names, xlabel=lab, title=title, xlim=(0.9, 1.8), ylim=(-0.55, len(names) - 0.5))
        ax.axvline(1, color='#666666', ls='--', lw=1)
        ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=9)
        ax.grid(axis='y', visible=False)
    axs[1].set_facecolor('#f6f8fa')
    fig.text(0.5, -0.015, '** В обеих регрессиях для каждого корпуса — одна выборка. Панели напрямую не сопоставляются.', ha='center', fontsize=8)
    save(fig, '13_weekend_robustness.png')
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True, layout='constrained')
    for ax, key, title in zip(axs, ('blog', 'blog_without_density'), ('С плотностью', 'Без плотности')):
        for i, c in enumerate(CORPORA):
            for k, g in enumerate(('personal', 'corporate')):
                r = cal[c][key][g]
                error(ax, r['ratio'], 1 - k + (0.13 if i == 0 else -0.13), r['ci95'], COLORS[i], label=LABELS[c] if k == 0 else None)
        ax.axvline(1, color='gray', ls='--')
        ax.set(title=title, yticks=[1, 0], yticklabels=['Личные', 'Корпоративные'], xlabel='Условное отношение', ylim=(-0.5, 1.5))
        ax.legend(frameon=False, fontsize=8)
    save(fig, '14_blog_interaction.png')
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4), layout='constrained')
    for ax, key, title in zip(axs, ('comments', 'favorites'), ('Комментарии', 'Закладки')):
        for i, c in enumerate(CORPORA):
            r = cal[c]['engagement'][key]
            ax.bar(i - 0.17, r['weekday'], 0.34, color='#a6adb4', label='Будни' if i == 0 else None)
            ax.bar(i + 0.17, r['weekend'], 0.34, color=COLORS[i], label='Выходные' if i == 0 else None)
        ax.set(xticks=[0, 1], xticklabels=list((LABELS[c] for c in CORPORA)), title=title, ylabel='Медиана на 1000 просмотров')
        ax.legend(frameon=False, fontsize=9)
    save(fig, '15_engagement.png')
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, layout='constrained')
    for i, (ax, c) in enumerate(zip(axs, CORPORA)):
        h = j['corpora'][c]['holiday']
        for k, window in enumerate(('0', '3', '7', '14')):
            r = h[window]
            v = r['diff_pp']
            ci = r['ci95_episode_week'] if k else old[c]['holiday']['pct_diff_pp_ci_author']
            ax.errorbar(k, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='o', capsize=3, color=COLORS[i], label='Локальная / месячная' if k == 0 else None)
            if k:
                ax.plot(k, r['matched_month_diff_pp'], 'D', color='#888888', ms=5, label='Месяц, те же статьи' if k == 1 else None)
        ax.axhline(0, color='gray', ls='--')
        ax.set(title=LABELS[c], xticks=range(4), xticklabels=['Месяц', '±3', '±7', '±14'], xlabel='Окно перцентиля, дни')
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel('Разница медианных перцентилей, п.п.')
    save(fig, '16_holiday_local.png')
    fig, axs = plt.subplots(2, 1, figsize=(9, 8.2), gridspec_kw={'height_ratios': [13, 4]}, layout='constrained')
    r = j['readability']['all']
    hubs = sorted(r['oborneva']['within_hubs'], key=lambda h: -r['oborneva']['within_hubs'][h]['n'])
    for i, (met, label) in enumerate(METRICS.items()):
        ax = axs[0]
        for k, h in enumerate(hubs):
            v = r[met]['within_hubs'][h]['rho']
            ax.plot(v, len(hubs) - k + (0.13 if i == 0 else -0.13), 'o', ms=4, color=COLORS[i], label=label if k == 0 else None)
        ax = axs[1]
        for k, threshold in enumerate(('2', '3')):
            a = r[met]['within_author'][threshold]['equal_author']
            error(ax, a['mean_rho'], 1 - k + (0.13 if i == 0 else -0.13), a['ci95'], COLORS[i], label=label if k == 0 else None)
    axs[0].set(yticks=range(len(hubs), 0, -1), yticklabels=[f"{HUBS.get(h, h)} (n={r['oborneva']['within_hubs'][h]['n']})" for h in hubs], title='А. Корреляции внутри отдельных хабов', xlabel='ρ Спирмена с VPD')
    axs[1].set(yticks=[1, 0], yticklabels=['≥2 статьи', '≥3 статьи'], title='Б. Средняя внутриавторская корреляция', xlabel='Среднее ρ (один автор — один голос)', ylim=(-0.6, 1.6))
    for ax in axs:
        ax.axvline(0, color='gray', ls='--')
        ax.legend(frameon=False, fontsize=9)
    save(fig, '17_readability_sensitivity.png')
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), layout='constrained')
    for i, (ax, c) in enumerate(zip(axs, CORPORA)):
        ages = j['corpora'][c]['fold_ages']
        keys = list(ages)
        for k, key in enumerate(keys):
            mn, q1, med, q3, mx = ages[key]['age']
            y = len(keys) - 1 - k
            ax.plot([mn, mx], [y, y], color=COLORS[i], alpha=0.65)
            ax.plot([q1, q3], [y, y], lw=9, color=COLORS[i], alpha=0.4, solid_capstyle='butt')
            ax.plot(med, y, 'o', color=COLORS[i], ms=5)
        ax.set(yticks=range(len(keys) - 1, -1, -1), yticklabels=[FOLDS[k] for k in keys], xscale='log', title=LABELS[c], xlabel='Возраст, дни')
    save(fig, '18_fold_age.png')
    fig, axs = plt.subplots(1, 2, figsize=(8.9, 3.9), sharey=True, layout='constrained')
    for ax, c in zip(axs, CORPORA):
        for key, label in POWER.items():
            r = cal[c]['power'][key]
            ax.plot([10, 20, 30, 40], list(r['per_arm'].values()), 'o-', label=label, ms=4)
        ax.set(title=LABELS[c], xlabel='Минимальный относительный прирост, %', yscale='log', xticks=[10, 20, 30, 40])
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel('Статей в каждой группе')
    save(fig, '19_power.png')
    short_density()

def short_density():
    """Short article density plot: data points, author CIs and weekend counts."""
    res = json.loads((RESULTS / 'part4_results.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(8.6, 4.5), layout='constrained')
    for i, c in enumerate(CORPORA):
        rows = res[c]['quintiles']
        xs = [row['q'] + (-0.15 if i == 0 else 0.15) for row in rows]
        for k, (x, row) in enumerate(zip(xs, rows)):
            y = row['ratio']
            lo, hi = row['ci95_author']
            ax.errorbar(x, y, yerr=[[max(y - lo, 0)], [max(hi - y, 0)]],
                        fmt='o', ms=6, capsize=4, lw=1.4, color=COLORS[i],
                        label=LABELS[c] if k == 0 else None)
            ax.annotate(f"n={row['n_weekend']}", (x, hi), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=COLORS[i])
    ax.axhline(1, color='#777777', ls='--', lw=1)
    ax.grid(axis='x', visible=False)
    ax.set(xticks=range(1, 6), xlim=(0.55, 5.45), ylim=(0.45, 4.8),
           xlabel='Группа плотности: 1 — самая редкая, 5 — самая плотная',
           ylabel='Отношение медиан VPD\nвыходных к будням',
           title='Выходные против будней при сходной плотности')
    ax.legend(loc='upper left', frameon=False, ncol=2)
    ax.text(0.99, 0.98, 'n — число выходных статей', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#555555')
    fig.savefig(BASE / 'short_article/figures/weekend_density.png',
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
