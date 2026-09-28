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

import {
  createContext,
  useContext,
  useMemo,
  useRef,
  useState,
  type PropsWithChildren,
} from 'react';
import type { AgentSubscriber } from '@/components/block/chat/types';
import type { FleetStatusResult, FleetWell } from '@/api/fleet/types';

const FLEET_TOOL_NAME = 'get_fleet_status';

interface FleetContextValue {
  // Newest get_fleet_status result, or null before any call.
  fleet: FleetWell[] | null;
  // The as_of the fleet was scored at (may be several questions ago).
  asOf: string | null;
  // True when the last agent turn answered without a fresh get_fleet_status.
  stale: boolean;
  // Subscriber to hand to <ChatProvider subscriber=...> so this panel can read
  // tool results out of the chat event stream.
  subscriber: AgentSubscriber;
}

const FleetContext = createContext<FleetContextValue | null>(null);

function parseResult(content: unknown): FleetStatusResult | null {
  // ToolCallResultEvent.content arrives as a string; parse defensively.
  try {
    if (typeof content === 'string') {
      return JSON.parse(content) as FleetStatusResult;
    }
    if (content && typeof content === 'object') {
      return content as FleetStatusResult;
    }
  } catch {
    return null;
  }
  return null;
}

export function FleetProvider({ children }: PropsWithChildren) {
  const [fleet, setFleet] = useState<FleetWell[] | null>(null);
  const [asOf, setAsOf] = useState<string | null>(null);
  const [stale, setStale] = useState(false);

  // Map toolCallId -> toolName, populated on start, consumed on result.
  const toolNamesById = useRef<Record<string, string>>({});
  // Did the current run call get_fleet_status? Reset on run start.
  const sawFleetThisRun = useRef(false);

  const subscriber = useMemo<AgentSubscriber>(
    () => ({
      onStepStartedEvent() {
        // A new agent turn begins; assume stale until a fleet result lands.
        sawFleetThisRun.current = false;
      },
      onToolCallStartEvent({ event }) {
        const anyEvent = event as { toolCallId?: string; toolCallName?: string };
        if (anyEvent.toolCallId && anyEvent.toolCallName) {
          toolNamesById.current[anyEvent.toolCallId] = anyEvent.toolCallName;
        }
      },
      onToolCallResultEvent({ event }) {
        const anyEvent = event as { toolCallId?: string; content?: unknown };
        const id = anyEvent.toolCallId ?? '';
        const name = toolNamesById.current[id];
        if (name !== FLEET_TOOL_NAME) {
          return;
        }
        const parsed = parseResult(anyEvent.content);
        if (!parsed || !parsed.fleet || parsed.fleet.length === 0) {
          return;
        }
        sawFleetThisRun.current = true;
        setFleet(parsed.fleet);
        setAsOf(parsed.fleet[0]?.as_of ?? null);
        setStale(false);
      },
      onRunFinishedEvent() {
        // If the turn answered without calling get_fleet_status, HOLD the
        // previous circles and mark them stale - never blank the map.
        if (!sawFleetThisRun.current && fleet) {
          setStale(true);
        }
      },
    }),
    [fleet]
  );

  const value = useMemo<FleetContextValue>(
    () => ({ fleet, asOf, stale, subscriber }),
    [fleet, asOf, stale, subscriber]
  );

  return <FleetContext.Provider value={value}>{children}</FleetContext.Provider>;
}

export function useFleet(): FleetContextValue {
  const ctx = useContext(FleetContext);
  if (!ctx) {
    throw new Error('useFleet must be used within a FleetProvider');
  }
  return ctx;
}

// Standalone subscriber factory for the rare case a caller needs it without the
// provider (kept for symmetry; the provider is the normal path).
export function useFleetSubscriber(): AgentSubscriber {
  return useFleet().subscriber;
}

export const _test = { parseResult };
