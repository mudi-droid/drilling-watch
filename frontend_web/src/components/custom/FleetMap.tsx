// Copyright 2026 DataRobot, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { MapContainer, TileLayer, Circle, Popup, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useFleet } from '@/api/fleet/fleet-context';
import { MAP_CENTRE, MAP_LABEL, MAP_ZOOM, WELL_POSITIONS, styleForState } from '@/api/fleet/wells';
import type { FleetWell } from '@/api/fleet/types';

// A circle radius in metres that reads well at zoom 9 on this block.
const CIRCLE_RADIUS_M = 2600;

export function FleetMap() {
  const { fleet, asOf, stale } = useFleet();

  // Index the newest scores by well so static positions drive placement and
  // the fleet result only drives colour and numbers.
  const byWell = new Map<string, FleetWell>();
  for (const w of fleet ?? []) {
    byWell.set(w.well, w);
  }

  return (
    <div className="relative h-full min-h-0 w-full min-w-0">
      <MapContainer
        center={MAP_CENTRE}
        zoom={MAP_ZOOM}
        scrollWheelZoom={false}
        className="h-full w-full"
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {WELL_POSITIONS.map(pos => {
          const scored = byWell.get(pos.well);
          const style = styleForState(scored?.state);
          const opacity = stale ? 0.4 : 0.85;
          return (
            <Circle
              key={pos.well}
              center={[pos.latitude, pos.longitude]}
              radius={CIRCLE_RADIUS_M}
              pathOptions={{
                color: style.stroke,
                fillColor: style.fill,
                fillOpacity: opacity,
                opacity,
                weight: 2,
              }}
            >
              <Tooltip permanent direction="top" offset={[0, -6]}>
                {pos.well}
              </Tooltip>
              <Popup>
                <div className="text-sm">
                  <div className="font-semibold">{pos.well}</div>
                  <div>state: {scored ? scored.state : 'not scored'}</div>
                  {scored && (
                    <>
                      <div>probability: {scored.probability.toFixed(3)}</div>
                      <div>delta flow: {scored.delta_flow_gpm} gpm</div>
                      <div>pit rate: {scored.pit_rate_bbl_per_min} bbl/min</div>
                      <div>SPP: {scored.spp_psi} psi</div>
                      <div>as of: {scored.as_of}</div>
                    </>
                  )}
                </div>
              </Popup>
            </Circle>
          );
        })}
      </MapContainer>

      {/* SHOW THE as_of ON THE MAP ITSELF - the build is stateless, so the map
          shows whatever moment the last tool call used, never "now". */}
      <div className="pointer-events-none absolute top-2 left-2 z-[1000] rounded bg-background/85 px-2 py-1 text-xs text-foreground shadow">
        <div className="font-medium">{MAP_LABEL}</div>
        <div>
          {asOf ? `as of ${asOf}` : 'not yet scored - ask "what\u2019s the fleet doing?"'}
          {stale && asOf ? ' (stale)' : ''}
        </div>
      </div>
    </div>
  );
}
