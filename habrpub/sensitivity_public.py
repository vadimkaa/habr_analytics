"""Publication tables and figures derived from the validated sensitivity snapshot."""
import json
from . import RESULTS
LABELS = {'ai': 'ИИ/МЛ', 'wide': 'Пять хабов', 'all': 'Вся выборка'}
HUBS = {'artificial_intelligence': 'Искусственный интеллект', 'infosecurity': 'Информационная безопасность', 'popular_science': 'Научно-популярное', 'machine_learning': 'Машинное обучение', 'programming': 'Программирование', 'python': 'Python', 'futurenow': 'Будущее здесь', 'webdev': 'Веб-разработка', 'open_source': 'Open source', 'read': 'Читальный зал', 'maths': 'Математика', 'it-infrastructure': 'IT-инфраструктура', 'bothub': 'bothub'}
CORE_HUBS = {'artificial_intelligence', 'infosecurity', 'popular_science', 'machine_learning', 'programming', 'python', 'webdev'}
FOLDS = {'dev_train': 'Обучение', 'validation': 'Валидация', 'extra_check': 'Доп. проверка', 'audit_exposed': 'Итоговая проверка', 'final_train': 'Финальное обучение'}
METRICS = {'oborneva': 'ИУФО', 'long_word_share': 'Доля длинных слов'}
POWER = {'raw': 'Исходный разброс', 'adjusted': 'Остатки OLS', 'within_author': 'Внутриавторские остатки'}
CORPORA = ('ai', 'wide')

def snapshot():
    return json.loads((RESULTS / '17_final_sensitivity.json').read_text(encoding='utf-8'))

def values_sensitivity(R, f, interval, table):
    J = snapshot()
    C = {c: J['corpora'][c]['calendar'] for c in CORPORA}
    H = {c: J['corpora'][c]['holiday'] for c in CORPORA}
    read = J['readability']
    v = {}

    def fmt(x, n=3):
        return '—' if x is None else f(x, n)

    def ci(x, n=2):
        return '—' if x is None else interval(x, n)

    def estimate(x, n=2):
        return f(x['ratio'], n) + (f" {ci(x['ci95'], n)}" if x.get('ci95') is not None else '')
    v['s_author_iqr_table'] = table(['Корпус', 'Минимум статей', '25-й перцентиль', 'Медиана', '75-й перцентиль'], [[LABELS[c], k, *[f(x, 1) for x in C[c]['author_iqr'][k]]] for c in CORPORA for k in ('3', '5')])
    v['s_age_window_table'] = table(['Возраст, дней', 'ИИ/МЛ: отношение [95% ДИ]; n вых./буд.', 'Пять хабов: отношение [95% ДИ]; n вых./буд.'], [[f'{lo}–{lo + 14}', *[f"{estimate(C[c]['age_windows'][str(lo)])}; {C[c]['age_windows'][str(lo)]['n_weekend']}/{C[c]['age_windows'][str(lo)]['n_weekday']}" for c in CORPORA]] for lo in range(30, 360, 15)])
    v['s_spec_table'] = table(['Спецификация', 'ИИ/МЛ', 'Пять хабов'], [[label, *[estimate(C[c][key], 3) for c in CORPORA]] for key, label in [('ols', 'Основная OLS'), ('spline', 'OLS: сплайн возраста'), ('quantile_views', 'Медианная: просмотры + 1'), ('quantile_vpd', 'Медианная: VPD (алгебраический контроль)')]])
    v['s_fe_table'] = table(['Спецификация', 'ИИ/МЛ', 'Пять хабов'], [[label, *[f"{estimate(C[c][key])}; n = {C[c][key]['n']}" for c in CORPORA]] for key, label in [('ols', 'Основная OLS (вся выборка)'), ('same_sample_ols', 'OLS на подвыборке FE'), ('fe', 'Авторские фиксированные эффекты')]] + [['Авторы в подвыборке FE', *[str(C[c]['fe']['n_authors']) for c in CORPORA]]])
    v['s_blog_table'] = table(['Корпус', 'Спецификация', 'Личные', 'Корпоративные', 'Отношение отношений', 'p'], [[LABELS[c], label, *[estimate(C[c][key][k]) for k in ['personal', 'corporate', 'ratio_of_ratios']], f(C[c][key]['ratio_of_ratios']['p'], 3)] for c in CORPORA for key, label in [('blog', 'С плотностью'), ('blog_without_density', 'Без плотности')]])
    v['s_engagement_table'] = table(['Корпус', 'Метрика', 'Будни', 'Выходные', 'Разница [95% ДИ]', 'Отношение сумм: будни / выходные'], [[LABELS[c], label, f(r['weekday']), f(r['weekend']), f"{f(r['weekend'] - r['weekday'])} {ci(r['diff_ci95'])}", f"{f(r['aggregate_weekday'])} / {f(r['aggregate_weekend'])}"] for c in CORPORA for key, label in [('comments', 'Комментарии'), ('favorites', 'Закладки')] for r in [C[c]['engagement'][key]]])
    v['s_density_table'] = table(['Корпус', 'Дни', 'n', 'ρ с VPD', 'Плотность Q25 / Q50 / Q75', 'Отношение на +10 статей [95% ДИ]'], [[LABELS[c], label, str(r['raw']['n']), f(r['raw']['rho'], 3), ' / '.join((f(x, 0) for x in r['density_iqr'])), estimate(r['adjusted_per10'], 3)] for c in CORPORA for key, label in [('weekday', 'Будни'), ('weekend', 'Выходные')] for r in [C[c]['density'][key]]])
    v['s_holiday_table'] = table(['Корпус', 'Окно ранга', 'Праздн. / контр. / эпиз.', 'Разница, п.п. [95% ДИ]', 'Месячная оценка на тех же статьях', 'Медиана размера окна'], [[LABELS[c], '±' + k + ' дн.' if k != '0' else 'Месяц', f"{r['n_holiday']} / {r['n_control']} / {r['n_episodes']}", f"{f(r['diff_pp'], 1)} {ci(r['ci95_episode_week'], 1)}", f(r['matched_month_diff_pp'], 1), fmt(r['median_reference_n'], 0)] for c in CORPORA for k in ('7', '3', '14', '0') for r in [H[c][k]]])
    rr = read['all']
    hub = rr['oborneva']['within_hubs']
    v['s_hub_table'] = table(['Хаб', 'В составе корпусов', 'n', 'ИУФО: ρ', 'Длинные слова: ρ'], [[HUBS.get(h, h), 'Да' if h in CORE_HUBS else 'Нет', str(r['n']), f(r['rho'], 3), f(rr['long_word_share']['within_hubs'][h]['rho'], 3)] for h, r in sorted(hub.items(), key=lambda a: (-a[1]['n'], a[0]))])
    v['s_read_within_table'] = table(['Корпус', 'Метрика', 'Без поправок', 'Частная ρ', 'Внутри автора ≥2', 'Наклон ≥2 [ДИ]', 'Внутри автора ≥3', 'Статей / авторов ≥2'], [[LABELS[c], label, f(r['raw'], 3), f(r['partial'], 3), fmt(a['adjusted']), f"{fmt(a['slope'])} {ci(a['slope_ci95'], 3)}" if a['slope'] is not None else '—', fmt(r['within_author']['3']['adjusted']), f"{a['n']}/{a['authors']}"] for c in ('all', *CORPORA) for met, label in METRICS.items() for r in [read[c][met]] for a in [r['within_author']['2']]])

    def eq(r):
        return f"{f(r['mean_rho'], 3)} {ci(r['ci95'], 3)}; n = {r['n_authors']}"
    v['s_read_equal_table'] = table(['Корпус', 'Метрика', '≥2 статьи: средняя ρ [ДИ]; авторов', '≥3 статьи: средняя ρ [ДИ]; авторов'], [[LABELS[c], label, *[eq(read[c][met]['within_author'][k]['equal_author']) for k in ('2', '3')]] for c in ('all', *CORPORA) for met, label in METRICS.items()])
    v['s_fold_table'] = table(['Корпус', 'Выборка', 'n', 'Возраст: min / Q25 / медиана / Q75 / max, дней', 'Медиана просмотров'], [[LABELS[c], FOLDS[k], str(r['n']), ' / '.join((f(x, 1) for x in r['age'])), f(r['median_views'], 0)] for c in CORPORA for k, r in J['corpora'][c]['fold_ages'].items()])
    v['s_power_table'] = table(['Корпус', 'Сценарий разброса', 'σ', '+10%', '+20%', '+30%', '+40%'], [[LABELS[c], label, f(r['sigma'], 3), *[str(r['per_arm'][k]) for k in ('1.1', '1.2', '1.3', '1.4')]] for c in CORPORA for key, label in POWER.items() for r in [C[c]['power'][key]]])
    for c in CORPORA:
        for k in ('3', '7', '14', '0'):
            r = H[c][k]
            v[f's_holiday_{k}_{c}'] = f(r['diff_pp'], 1)
            v[f's_holiday_{k}_ci_{c}'] = ci(r['ci95_episode_week'], 1)
            v[f's_holiday_{k}_matched_{c}'] = f(r['matched_month_diff_pp'], 1)
        for k in ('fe', 'same_sample_ols', 'ols', 'spline', 'quantile_views', 'quantile_vpd'):
            r = C[c][k]
            v[f's_{k}_{c}'] = f(r['ratio'])
            v[f's_{k}_ci_{c}'] = ci(r['ci95'])
        v[f's_icc_{c}'] = f(C[c]['random_intercept']['icc'] * 100, 1)
        for k in ('personal', 'corporate', 'ratio_of_ratios'):
            r = C[c]['blog'][k]
            v[f's_{k}_{c}'] = f(r['ratio'])
            v[f's_{k}_ci_{c}'] = ci(r['ci95'])
        for k in ('final_train', 'audit_exposed'):
            v[f's_age_{k}_{c}'] = f(J['corpora'][c]['fold_ages'][k]['age'][2], 0)
    for c in ('all', *CORPORA):
        for met in METRICS:
            r = read[c][met]
            v[f's_partial_{met}_{c}'] = f(r['partial'], 3)
            for k in ('2', '3'):
                v[f's_within_{met}_{k}_{c}'] = fmt(r['within_author'][k]['adjusted'])
                v[f's_equal_{met}_{k}_{c}'] = f(r['within_author'][k]['equal_author']['mean_rho'], 3)
    return {key: v[key] for key in ('s_age_audit_exposed_ai', 's_age_audit_exposed_wide', 's_age_final_train_ai', 's_age_final_train_wide', 's_age_window_table', 's_author_iqr_table', 's_blog_table', 's_corporate_ci_wide', 's_corporate_wide', 's_density_table', 's_engagement_table', 's_equal_long_word_share_2_all', 's_equal_oborneva_2_all', 's_fe_ai', 's_fe_ci_ai', 's_fe_ci_wide', 's_fe_table', 's_fe_wide', 's_fold_table', 's_holiday_14_ai', 's_holiday_14_wide', 's_holiday_3_ai', 's_holiday_3_matched_ai', 's_holiday_3_matched_wide', 's_holiday_3_wide', 's_holiday_7_ai', 's_holiday_7_ci_ai', 's_holiday_7_ci_wide', 's_holiday_7_wide', 's_holiday_table', 's_hub_table', 's_icc_ai', 's_icc_wide', 's_partial_long_word_share_all', 's_partial_oborneva_all', 's_personal_ci_wide', 's_personal_wide', 's_power_table', 's_ratio_of_ratios_ci_wide', 's_ratio_of_ratios_wide', 's_read_equal_table', 's_read_within_table', 's_same_sample_ols_ai', 's_same_sample_ols_ci_ai', 's_same_sample_ols_ci_wide', 's_same_sample_ols_wide', 's_spec_table', 's_within_long_word_share_2_ai', 's_within_long_word_share_2_all', 's_within_long_word_share_3_all', 's_within_oborneva_2_ai', 's_within_oborneva_2_all', 's_within_oborneva_2_wide', 's_within_oborneva_3_all')}
