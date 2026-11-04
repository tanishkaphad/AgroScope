import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Eye, Layers, Maximize2, Navigation, ZoomIn } from 'lucide-react';

const CARTO_API_KEY = import.meta.env.VITE_CARTO_API_KEY;

const BASE_LAYERS = {
  street: CARTO_API_KEY
    ? {
        url: `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${CARTO_API_KEY}`,
        options: {
          subdomains: 'abcd',
          maxNativeZoom: 18,
          maxZoom: 18,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attribution/">CARTO</a>',
        },
      }
    : {
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        options: {
          maxZoom: 18,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        },
      },
  // Free for non-commercial use only. Commercial use needs a paid EOX licence.
  satellite: {
    url: 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg',
    options: {
      maxNativeZoom: 14,
      maxZoom: 18,
      attribution: 'EOxCloudless <a href="https://cloudless.eox.at">https://cloudless.eox.at</a> by <a href="https://eox.at/">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2025)',
    },
  },
};

// Colours chosen to stay readable on both the light street map and the dark satellite map.
const STRESS_COLORS = { HIGH: '#dc2626', MODERATE: '#d97706', LOW: '#059669' };
const PRIORITY_COLORS = { 1: '#dc2626', 2: '#d97706', 3: '#2563eb' };
const BOUNDARY_COLORS = { street: '#0f4c5c', satellite: '#7dd3fc' };
const SELECTED_OUTLINE = { street: '#111827', satellite: '#ffffff' };
const HEADWORKS = [18.1482, 74.0485];

function escapeHtml(text) {
  const element = document.createElement('div');
  element.textContent = text;
  return element.innerHTML;
}

function LegendRow({ chip, label, value }) {
  return (
    <div className="legend-row">
      <div className="legend-chip">
        <span className={`chip-color ${chip}`}></span>
        <span>{label}</span>
      </div>
      <span className="legend-value">{value}</span>
    </div>
  );
}

export default function MapContainer({
  studyAreaGeoJson,
  stressMapGeoJson,
  irrigationMapGeoJson,
  stats,
  isLoading,
  activeLayer = 'stress',
  onLayerChange,
  onSelectParcel,
  selectedParcelId,
  t,
}) {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const boundaryLayerRef = useRef(null);
  const dataLayerRef = useRef(null);
  const headworksRef = useRef(null);

  const [baseMap, setBaseMap] = useState('street'); // 'street' | 'satellite'

  const stress = stats?.stress_breakdown;
  const priority = stats?.irrigation_priority_breakdown;
  const legendValue = (pct, hectares) => (
    stats ? `${pct}% (${Math.round(hectares).toLocaleString('en-US')} ${t.unitHa})` : t.noData
  );

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [18.18, 74.45],
      zoom: 10,
      zoomControl: false,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);
    map.attributionControl.addAttribution('Weather: <a href="https://open-meteo.com/">Open-Meteo</a>');

    const baseTile = L.tileLayer(BASE_LAYERS.street.url, BASE_LAYERS.street.options).addTo(map);

    mapInstanceRef.current = map;
    mapInstanceRef.current._baseTile = baseTile;

    headworksRef.current = L.circleMarker(HEADWORKS, {
      radius: 8,
      fillColor: '#0f4c5c',
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.9,
    }).addTo(map);

    // The panel changes size with the layout, and Leaflet needs telling or tiles leave gaps.
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(mapRef.current);

    return () => {
      observer.disconnect();
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    headworksRef.current?.bindPopup(`<div class="map-popup"><strong>${escapeHtml(t.headworks)}</strong><br/>${escapeHtml(t.headworksDetail)}</div>`);
  }, [t]);

  // Handle Base Map Switch
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !map._baseTile) return;

    map.removeLayer(map._baseTile);
    const newTile = L.tileLayer(BASE_LAYERS[baseMap].url, BASE_LAYERS[baseMap].options).addTo(map);
    map._baseTile = newTile;
    newTile.bringToBack();
  }, [baseMap]);

  // Render Boundary Layer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !studyAreaGeoJson) return;

    if (boundaryLayerRef.current) {
      map.removeLayer(boundaryLayerRef.current);
    }

    try {
      const boundaryLayer = L.geoJSON(studyAreaGeoJson, {
        style: {
          color: BOUNDARY_COLORS.street,
          weight: 2.5,
          opacity: 0.9,
          fillOpacity: 0,
          dashArray: '4, 4',
        },
      }).addTo(map);

      boundaryLayerRef.current = boundaryLayer;

      const bounds = boundaryLayer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [25, 25] });
      }
    } catch (err) {
      console.warn('Failed to render boundary GeoJSON:', err);
    }
  }, [studyAreaGeoJson]);

  useEffect(() => {
    boundaryLayerRef.current?.setStyle({ color: BOUNDARY_COLORS[baseMap] });
  }, [baseMap, studyAreaGeoJson]);

  // Render Data Layer (Stress Map or Irrigation Priority)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (dataLayerRef.current) {
      map.removeLayer(dataLayerRef.current);
    }

    const geoData = activeLayer === 'stress' ? stressMapGeoJson : irrigationMapGeoJson;
    if (!geoData || !geoData.features || geoData.features.length === 0) return;

    try {
      const dataLayer = L.geoJSON(geoData, {
        style: (feature) => {
          const props = feature.properties || {};
          const color = activeLayer === 'priority'
            ? PRIORITY_COLORS[props.priority] ?? PRIORITY_COLORS[2]
            : STRESS_COLORS[props.stress_class] ?? STRESS_COLORS.MODERATE;
          const isSelected = props.cell_id === selectedParcelId;

          return {
            color: isSelected ? SELECTED_OUTLINE[baseMap] : color,
            weight: isSelected ? 3 : 0.8,
            opacity: isSelected ? 1 : 0.8,
            fillColor: color,
            fillOpacity: isSelected ? 0.95 : 0.72,
          };
        },
        onEachFeature: (feature, layer) => {
          const props = feature.properties || {};
          const ndvi = props.ndvi !== undefined ? props.ndvi.toFixed(2) : '--';
          const ndwi = props.ndwi !== undefined ? props.ndwi.toFixed(2) : '--';
          const detail = activeLayer === 'priority'
            ? t[`priority${props.priority}`]
            : `${t.popupStress}: ${t.stress[props.stress_class] ?? props.stress_class}`;

          layer.on({
            click: (e) => {
              L.DomEvent.stopPropagation(e);
              if (onSelectParcel) {
                onSelectParcel(props);
              }
            },
            mouseover: () => {
              layer.setStyle({ weight: 2.5, opacity: 1, fillOpacity: 0.9 });
            },
            mouseout: () => {
              if (props.cell_id !== selectedParcelId) {
                dataLayer.resetStyle(layer);
              }
            },
          });

          layer.bindPopup(`
            <div class="map-popup">
              <strong>${escapeHtml(props.cell_id ?? '')}</strong><br/>
              <span>${escapeHtml(detail)}</span><br/>
              <span>NDVI ${ndvi}, NDWI ${ndwi}</span><br/>
              <span class="map-popup-hint">${escapeHtml(t.popupDetails)}</span>
            </div>
          `);
        },
      }).addTo(map);

      dataLayerRef.current = dataLayer;
    } catch (err) {
      console.warn('Failed to render data layer GeoJSON:', err);
    }
  }, [stressMapGeoJson, irrigationMapGeoJson, activeLayer, selectedParcelId, onSelectParcel, baseMap, t]);

  const fitCommandArea = () => {
    if (mapInstanceRef.current && boundaryLayerRef.current) {
      mapInstanceRef.current.fitBounds(boundaryLayerRef.current.getBounds(), { padding: [25, 25] });
    }
  };

  const focusHeadworks = () => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo(HEADWORKS, 13);
    }
  };

  return (
    <div className="map-panel" id="gis-map-panel">
      <div className="map-toolbar">
        <div className="map-layer-toggles" role="group" aria-label={t.layerGroup}>
          <button
            type="button"
            className={`layer-btn ${activeLayer === 'stress' ? 'active' : ''}`}
            aria-pressed={activeLayer === 'stress'}
            onClick={() => onLayerChange('stress')}
            id="toggle-layer-stress"
          >
            <Layers size={14} aria-hidden="true" />
            <span>{t.layerStress}</span>
          </button>
          <button
            type="button"
            className={`layer-btn ${activeLayer === 'priority' ? 'active' : ''}`}
            aria-pressed={activeLayer === 'priority'}
            onClick={() => onLayerChange('priority')}
            id="toggle-layer-priority"
          >
            <Navigation size={14} aria-hidden="true" />
            <span>{t.layerPriority}</span>
          </button>
        </div>

        <div className="map-quick-actions">
          <button
            type="button"
            className="layer-btn"
            onClick={() => setBaseMap(baseMap === 'street' ? 'satellite' : 'street')}
            title={t.baseMapHelp}
            id="toggle-basemap-btn"
          >
            <Eye size={14} aria-hidden="true" />
            <span>{baseMap === 'street' ? t.showSatellite : t.showStreet}</span>
          </button>
          <button type="button" className="icon-btn" onClick={fitCommandArea} title={t.fitArea} aria-label={t.fitArea} id="fit-boundary-btn">
            <Maximize2 size={15} aria-hidden="true" />
          </button>
          <button type="button" className="icon-btn" onClick={focusHeadworks} title={t.focusHeadworks} aria-label={t.focusHeadworks} id="focus-headworks-btn">
            <ZoomIn size={15} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="map-stage">
        <div id="leaflet-map-root" ref={mapRef} role="region" aria-label={t.mapRegion}></div>
        {isLoading && (
          <div className="map-loading skeleton" id="map-loading" role="status">
            <span>{t.loadingMap}</span>
          </div>
        )}
      </div>

      <div className="map-legend-floating" id="map-legend">
        <div className="legend-title">{activeLayer === 'stress' ? t.legendStress : t.legendPriority}</div>
        {activeLayer === 'stress' ? (
          <>
            <LegendRow chip="green" label={t.lowStress} value={legendValue(stress?.low_stress_pct, stress?.low_stress_hectares)} />
            <LegendRow chip="yellow" label={t.moderateStress} value={legendValue(stress?.moderate_stress_pct, stress?.moderate_stress_hectares)} />
            <LegendRow chip="red" label={t.highStress} value={legendValue(stress?.high_stress_pct, stress?.high_stress_hectares)} />
          </>
        ) : (
          <>
            <LegendRow chip="red" label={t.priority1} value={legendValue(priority?.priority_1_pct, priority?.priority_1_urgent_ha)} />
            <LegendRow chip="yellow" label={t.priority2} value={legendValue(priority?.priority_2_pct, priority?.priority_2_scheduled_ha)} />
            <LegendRow chip="blue" label={t.priority3} value={legendValue(priority?.priority_3_pct, priority?.priority_3_normal_ha)} />
          </>
        )}
      </div>
    </div>
  );
}
