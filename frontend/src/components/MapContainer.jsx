import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Eye, Layers, Maximize2, Navigation, ZoomIn } from 'lucide-react';

export default function MapContainer({
  studyAreaGeoJson,
  stressMapGeoJson,
  irrigationMapGeoJson,
  stats,
  activeLayer = 'stress',
  onLayerChange,
  onSelectParcel,
  selectedParcelId,
  currentLang = 'en',
}) {
  const mapRef = useRef(null);
  const stress = stats?.stress_breakdown;
  const legendValue = (pct, hectares) => (
    stress ? `${pct}% (${Math.round(hectares).toLocaleString()} ha)` : 'No data'
  );
  const mapInstanceRef = useRef(null);
  const boundaryLayerRef = useRef(null);
  const dataLayerRef = useRef(null);
  const selectedFeatureLayerRef = useRef(null);

  const [baseMap, setBaseMap] = useState('dark'); // 'dark' | 'satellite'

  const tileLayers = {
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  };

  const tileAttributions = {
    dark: '&copy; OpenStreetMap contributors &copy; CARTO',
    satellite: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
  };

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [18.18, 74.45],
      zoom: 10,
      zoomControl: false,
      attributionControl: false,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);

    // Initial base layer
    const baseTile = L.tileLayer(tileLayers.dark, {
      maxZoom: 18,
      attribution: tileAttributions.dark,
    }).addTo(map);

    mapInstanceRef.current = map;
    mapInstanceRef.current._baseTile = baseTile;

    // Headworks Marker (Veer Dam)
    const headworksMarker = L.circleMarker([18.1482, 74.0485], {
      radius: 8,
      fillColor: '#00f2fe',
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.9,
    }).addTo(map);

    headworksMarker.bindPopup(`
      <div style="font-family: sans-serif; font-size: 12px; color: #fff;">
        <strong style="color: #00f2fe;">Veer Dam Headworks</strong><br/>
        Nira Left Bank Canal Offtake<br/>
        <em>Krishna-Bhima Sub-basin</em>
      </div>
    `);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Handle Base Map Switch
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !map._baseTile) return;

    map.removeLayer(map._baseTile);
    const newTile = L.tileLayer(tileLayers[baseMap], {
      maxZoom: 18,
      attribution: tileAttributions[baseMap],
    }).addTo(map);
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
          color: '#00f2fe',
          weight: 2.5,
          opacity: 0.85,
          fillColor: '#00f2fe',
          fillOpacity: 0.04,
          dashArray: '4, 4',
        },
      }).addTo(map);

      boundaryLayerRef.current = boundaryLayer;

      // Fit bounds initially
      const bounds = boundaryLayer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [25, 25] });
      }
    } catch (err) {
      console.warn('Failed to render boundary GeoJSON:', err);
    }
  }, [studyAreaGeoJson]);

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
          let color = props.fill_color || '#ffc107';

          if (activeLayer === 'priority') {
            const p = props.priority || 2;
            color = p === 1 ? '#dc2626' : p === 2 ? '#f59e0b' : '#3b82f6';
          } else {
            const sc = props.stress_class || 'MODERATE';
            color = sc === 'HIGH' ? '#ef4444' : sc === 'LOW' ? '#10b981' : '#f59e0b';
          }

          const isSelected = props.cell_id === selectedParcelId;

          return {
            color: isSelected ? '#ffffff' : color,
            weight: isSelected ? 3 : 0.8,
            opacity: isSelected ? 1 : 0.65,
            fillColor: color,
            fillOpacity: isSelected ? 0.9 : 0.6,
          };
        },
        onEachFeature: (feature, layer) => {
          const props = feature.properties || {};
          const cid = props.cell_id || 'Parcel';
          const ndvi = props.ndvi !== undefined ? props.ndvi.toFixed(2) : '--';
          const ndwi = props.ndwi !== undefined ? props.ndwi.toFixed(2) : '--';
          const stressClass = props.stress_class || 'MODERATE';

          layer.on({
            click: (e) => {
              L.DomEvent.stopPropagation(e);
              if (onSelectParcel) {
                onSelectParcel(props);
              }
            },
            mouseover: () => {
              layer.setStyle({ weight: 2.5, opacity: 1, fillOpacity: 0.85 });
            },
            mouseout: () => {
              if (props.cell_id !== selectedParcelId) {
                dataLayer.resetStyle(layer);
              }
            },
          });

          // Lightweight popup
          layer.bindPopup(`
            <div style="font-family: sans-serif; font-size: 12px; line-height: 1.4;">
              <strong style="color: #00f2fe; font-size: 13px;">${cid}</strong><br/>
              <span>Stress: <strong>${stressClass}</strong></span><br/>
              <span>NDVI: ${ndvi} | NDWI: ${ndwi}</span><br/>
              <span style="color: #94a3b8; font-size: 11px;">Click for details</span>
            </div>
          `);
        },
      }).addTo(map);

      dataLayerRef.current = dataLayer;
    } catch (err) {
      console.warn('Failed to render data layer GeoJSON:', err);
    }
  }, [stressMapGeoJson, irrigationMapGeoJson, activeLayer, selectedParcelId, onSelectParcel]);

  const fitCommandArea = () => {
    if (mapInstanceRef.current && boundaryLayerRef.current) {
      mapInstanceRef.current.fitBounds(boundaryLayerRef.current.getBounds(), { padding: [25, 25] });
    }
  };

  const focusHeadworks = () => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([18.1482, 74.0485], 13);
    }
  };

  return (
    <div className="map-panel" id="gis-map-panel">
      {/* Map Toolbar */}
      <div className="map-toolbar">
        <div className="map-layer-toggles">
          <button
            type="button"
            className={`layer-btn ${activeLayer === 'stress' ? 'active' : ''}`}
            onClick={() => onLayerChange('stress')}
            id="toggle-layer-stress"
          >
            <Layers size={14} />
            <span>Water Stress</span>
          </button>
          <button
            type="button"
            className={`layer-btn ${activeLayer === 'priority' ? 'active' : ''}`}
            onClick={() => onLayerChange('priority')}
            id="toggle-layer-priority"
          >
            <Navigation size={14} />
            <span>Irrigation Priority</span>
          </button>
        </div>

        <div className="map-quick-actions">
          <button
            type="button"
            className="layer-btn"
            onClick={() => setBaseMap(baseMap === 'dark' ? 'satellite' : 'dark')}
            title="Toggle Base Tile"
            id="toggle-basemap-btn"
          >
            <Eye size={14} />
            <span>{baseMap === 'dark' ? 'Satellite' : 'Dark Carto'}</span>
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={fitCommandArea}
            title="Fit Command Area Boundary"
            id="fit-boundary-btn"
          >
            <Maximize2 size={15} />
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={focusHeadworks}
            title="Focus Veer Dam Headworks"
            id="focus-headworks-btn"
          >
            <ZoomIn size={15} />
          </button>
        </div>
      </div>

      {/* Leaflet Map Root */}
      <div id="leaflet-map-root" ref={mapRef}></div>

      {/* Floating Legend */}
      <div className="map-legend-floating" id="map-legend">
        <div className="legend-title">
          {activeLayer === 'stress' ? 'Crop Water Stress' : 'Irrigation Scheduling'}
        </div>
        {activeLayer === 'stress' ? (
          <>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color green"></span>
                <span>Low Stress</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>{legendValue(stress?.low_stress_pct, stress?.low_stress_hectares)}</span>
            </div>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color yellow"></span>
                <span>Moderate Stress</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>{legendValue(stress?.moderate_stress_pct, stress?.moderate_stress_hectares)}</span>
            </div>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color red"></span>
                <span>High Stress</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>{legendValue(stress?.high_stress_pct, stress?.high_stress_hectares)}</span>
            </div>
          </>
        ) : (
          <>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color red"></span>
                <span>Priority 1 (Urgent 24-48h)</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>Canal Priority</span>
            </div>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color yellow"></span>
                <span>Priority 2 (Next 3-5 Days)</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>Roster Turn</span>
            </div>
            <div className="legend-row">
              <div className="legend-chip">
                <span className="chip-color blue"></span>
                <span>Priority 3 (Adequate)</span>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>Routine</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
