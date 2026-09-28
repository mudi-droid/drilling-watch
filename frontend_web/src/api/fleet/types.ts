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

// The shape returned by the agent's get_fleet_status tool (one entry per well).
export interface FleetWell {
  well: string;
  as_of: string;
  probability: number;
  well_event: boolean;
  incident: 'kick' | 'lost_circulation' | null;
  state: string;
  delta_flow_gpm: number;
  pit_rate_bbl_per_min: number;
  spp_psi: number;
  gas_units: number;
  latitude: number;
  longitude: number;
}

export interface FleetStatusResult {
  fleet?: FleetWell[];
  error?: string;
}
