# Rapport d'Analyse d'Optimisation de Stratégie

## Résumé Exécutif
Analyse basée sur **{{ metrics.total_configs }}** configurations testées.

### Métriques de Performance

#### Ratio de Sharpe
- Maximum: {{ format_number(metrics.sharpe_stats.max) }}
- Moyenne: {{ format_number(metrics.sharpe_stats.mean) }}
- Écart-type: {{ format_number(metrics.sharpe_stats.std) }}

#### Rendement Total
- Maximum: {{ format_percentage(metrics.return_stats.max) }}
- Moyenne: {{ format_percentage(metrics.return_stats.mean) }}
- Écart-type: {{ format_percentage(metrics.return_stats.std) }}

#### Drawdown Maximum
- Minimum: {{ format_percentage(metrics.drawdown_stats.min) }}
- Moyenne: {{ format_percentage(metrics.drawdown_stats.mean) }}
- Écart-type: {{ format_percentage(metrics.drawdown_stats.std) }}

### Configurations Optimales

{% for config_name, config in metrics.best_configs.items() %}
#### Configuration {{ config_name|title }}
- Seuil de dépeg: {{ format_number(config.depeg_threshold) }}
- Montant de trade: {{ format_number(config.trade_amount) }}
- Stop loss: {{ format_number(config.stop_loss) }}
- Take profit: {{ format_number(config.take_profit) }}

**Performances:**
- Ratio de Sharpe: {{ format_number(config.sharpe_ratio) }}
- Rendement Total: {{ format_percentage(config.total_return) }}
- Drawdown Maximum: {{ format_percentage(config.max_drawdown) }}
- Win Rate: {{ format_percentage(config.win_rate) }}

{% endfor %}

### Analyse des Paramètres
{% for param, correlations in metrics.parameter_correlations.items() %}
#### {{ param|title }}
- Impact sur Sharpe: {{ format_number(correlations.sharpe_ratio) }}
- Impact sur Rendement: {{ format_number(correlations.total_return) }}
- Impact sur Drawdown: {{ format_number(correlations.max_drawdown) }}
{% endfor %}

## Recommandations

1. **Paramètre Clé**
   {% set max_impact = {'param': '', 'value': 0} %}
   {% for param, corr in metrics.parameter_correlations.items() %}
   {% if abs(corr.sharpe_ratio) > abs(max_impact.value) %}
   {% set _ = max_impact.update({'param': param, 'value': corr.sharpe_ratio}) %}
   {% endif %}
   {% endfor %}
   - Le paramètre le plus influent est `{{ max_impact.param }}`

2. **Observations**
   - Ratio de Sharpe moyen: {{ format_number(metrics.sharpe_stats.mean) }}
   - Drawdown maximum moyen: {{ format_percentage(metrics.drawdown_stats.mean) }}
   - Win Rate moyen: {{ format_percentage(metrics.win_rate_stats.mean) }}

_Rapport généré le {{ timestamp }}_