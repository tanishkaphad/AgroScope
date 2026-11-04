import React from 'react';
import { AlertTriangle, CheckCircle, Droplets, Flame, Layers } from 'lucide-react';

export default function KpiBar({ stats, currentLang = 'en' }) {
  if (!stats) return null;

  const labels = {
    en: {
      totalArea: 'Vegetated Area',
      highStress: 'High Water Stress',
      modStress: 'Moderate Stress',
      priority1: 'Priority 1 (Urgent 24-48h)',
      canopyVigor: 'Canopy Health (NDVI / NDWI)',
      cells: 'Active Parcels',
      statusGood: 'Optimal Soil Reserves',
      statusMon: 'Under Observation',
      statusUrgent: 'Canal Scheduling Required',
    },
    mr: {
      totalArea: 'एकूण वनस्पती क्षेत्र',
      highStress: 'तीव्र पाण्याचा ताण',
      modStress: 'मध्यम पाण्याचा ताण',
      priority1: 'तातडीचे सिंचन (२४-४८ तास)',
      canopyVigor: 'पिकांचा ओलावा (NDVI / NDWI)',
      cells: 'सक्रिय भूखंड',
      statusGood: 'समाधानकारक ओलावा',
      statusMon: 'निरीक्षणाखाली',
      statusUrgent: 'कालवा पाळीची गरज',
    },
    hi: {
      totalArea: 'कुल वनस्पति क्षेत्र',
      highStress: 'गंभीर जल तनाव',
      modStress: 'मध्यम जल तनाव',
      priority1: 'प्राथमिकता १ (तुरंत सिंचाई २४-४८ घंटे)',
      canopyVigor: 'फसल नमी सूचकांक (NDVI / NDWI)',
      cells: 'सक्रिय पार्सल',
      statusGood: 'उचित नमी उपलब्ध',
      statusMon: 'निगरानी में',
      statusUrgent: 'नहर जल की तत्काल आवश्यकता',
    },
  };

  const t = labels[currentLang] || labels.en;
  const stress = stats.stress_breakdown || {};
  const priority = stats.irrigation_priority_breakdown || {};
  const canopy = stats.mean_canopy_metrics || {};

  const totalHa = Number(stats.total_cultivated_hectares || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const highHa = Number(stress.high_stress_hectares || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const highPct = stress.high_stress_pct ?? 0.0;

  const modHa = Number(stress.moderate_stress_hectares || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const modPct = stress.moderate_stress_pct ?? 0;

  const p1Ha = Number(priority.priority_1_urgent_ha || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const meanNdvi = canopy.mean_ndvi ?? 0;
  const meanNdwi = canopy.mean_ndwi ?? 0;

  return (
    <section className="kpi-section" id="kpi-metrics-bar">
      <div className="kpi-grid">
        {/* Total Area */}
        <div className="kpi-card total-area" id="kpi-total-area">
          <div className="kpi-header">
            <span className="kpi-label">{t.totalArea}</span>
            <Layers className="kpi-icon" size={18} color="var(--brand-cyan)" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{totalHa}</span>
            <span className="kpi-unit">ha</span>
          </div>
          <div className="kpi-subtext">
            {stats.active_cultivated_cells?.toLocaleString()} {t.cells} (250m resolution)
          </div>
        </div>

        {/* High Stress */}
        <div className="kpi-card high-stress" id="kpi-high-stress">
          <div className="kpi-header">
            <span className="kpi-label">{t.highStress}</span>
            <Flame className="kpi-icon" size={18} color="#ef4444" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value" style={{ color: highPct > 10 ? '#ef4444' : '#f8fafc' }}>
              {highHa}
            </span>
            <span className="kpi-unit">ha ({highPct}%)</span>
          </div>
          <div className="kpi-subtext" style={{ color: highPct === 0 ? '#34d399' : '#f87171' }}>
            {highPct === 0 ? `✓ ${t.statusGood}` : `⚠ ${t.statusUrgent}`}
          </div>
        </div>

        {/* Moderate Stress */}
        <div className="kpi-card mod-stress" id="kpi-moderate-stress">
          <div className="kpi-header">
            <span className="kpi-label">{t.modStress}</span>
            <AlertTriangle className="kpi-icon" size={18} color="#f59e0b" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value" style={{ color: '#fbbf24' }}>
              {modHa}
            </span>
            <span className="kpi-unit">ha ({modPct}%)</span>
          </div>
          <div className="kpi-subtext">
            {t.statusMon}
          </div>
        </div>

        {/* Priority 1 / Canopy Metrics */}
        <div className="kpi-card low-stress" id="kpi-priority1-urgent">
          <div className="kpi-header">
            <span className="kpi-label">{t.canopyVigor}</span>
            <Droplets className="kpi-icon" size={18} color="#10b981" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value" style={{ color: '#34d399' }}>
              {meanNdvi}
            </span>
            <span className="kpi-unit">NDVI | {meanNdwi} NDWI</span>
          </div>
          <div className="kpi-subtext">
            {p1Ha > 0 ? `🚨 ${p1Ha} ha ${t.statusUrgent}` : `✓ ${p1Ha} ha in Priority 1 Urgent`}
          </div>
        </div>
      </div>
    </section>
  );
}
