import React from 'react';
import { AlertTriangle, Droplets, Flame, Info, Layers } from 'lucide-react';
import { formatNumber } from '../i18n';

function KpiCard({ id, variant, label, icon, value, unit, subtext, help, isLoading }) {
  return (
    <div className={`kpi-card ${variant}`} id={id} title={help} aria-busy={isLoading}>
      <div className="kpi-header">
        <div className="kpi-heading">
          <span className="kpi-icon-badge">{icon}</span>
          <span className="kpi-label">{label}</span>
        </div>
      </div>
      {isLoading ? (
        <>
          <div className="kpi-value-row">
            <span className="skeleton skeleton-value" />
          </div>
          <div className="kpi-subtext">
            <span className="skeleton skeleton-text" style={{ width: '70%' }} />
          </div>
        </>
      ) : (
        <div className="fade-in">
          <div className="kpi-value-row">
            <span className="kpi-value">{value}</span>
            {unit && <span className="kpi-unit">{unit}</span>}
          </div>
          <div className="kpi-subtext">{subtext}</div>
        </div>
      )}
    </div>
  );
}

export default function KpiBar({ stats, isLoading, t }) {
  const stress = stats?.stress_breakdown ?? {};
  const priority = stats?.irrigation_priority_breakdown ?? {};
  const canopy = stats?.mean_canopy_metrics ?? {};
  const hectares = (value) => (stats ? formatNumber(value ?? 0) : '--');
  const unit = stats ? t.unitHa : null;
  const share = (pct) => (stats ? t.ofVegetated(pct ?? 0) : t.notAvailable);

  return (
    <section className="kpi-section" id="kpi-metrics-bar">
      <div className="kpi-grid">
        <KpiCard
          id="kpi-total-area"
          variant="total-area"
          label={t.vegetatedArea}
          icon={<Layers className="kpi-icon" size={18} aria-hidden="true" />}
          value={hectares(stats?.total_cultivated_hectares)}
          unit={unit}
          subtext={stats ? (
            <span className="kpi-note">
              {t.cellsAt250m(formatNumber(stats.active_cultivated_cells))}
              <Info size={12} aria-hidden="true" />
              <span className="visually-hidden">{t.vegetatedHelp}</span>
            </span>
          ) : t.notAvailable}
          help={t.vegetatedHelp}
          isLoading={isLoading}
        />
        <KpiCard
          id="kpi-high-stress"
          variant="high-stress"
          label={t.highStress}
          icon={<Flame className="kpi-icon" size={18} aria-hidden="true" />}
          value={hectares(stress.high_stress_hectares)}
          unit={unit}
          subtext={share(stress.high_stress_pct)}
          isLoading={isLoading}
        />
        <KpiCard
          id="kpi-moderate-stress"
          variant="mod-stress"
          label={t.moderateStress}
          icon={<AlertTriangle className="kpi-icon" size={18} aria-hidden="true" />}
          value={hectares(stress.moderate_stress_hectares)}
          unit={unit}
          subtext={share(stress.moderate_stress_pct)}
          isLoading={isLoading}
        />
        <KpiCard
          id="kpi-priority1-urgent"
          variant="priority-area"
          label={t.priority1Kpi}
          icon={<Droplets className="kpi-icon" size={18} aria-hidden="true" />}
          value={hectares(priority.priority_1_urgent_ha)}
          unit={unit}
          subtext={stats ? t.canopyLine(canopy.mean_ndvi, canopy.mean_ndwi) : t.notAvailable}
          isLoading={isLoading}
        />
      </div>
    </section>
  );
}
