'use client';
import React from 'react';
import {
  isToolInvocationPart,
  isMessageStateEvent,
  type ChatStateEvent,
  type ToolInvocationData,
} from '@/components/block/chat/types';
import { type MessageResponse } from '@/api/chat/types';

/**
 * Fleet panel — spike.
 *
 * Reads the most recent `get_fleet_status` tool result straight out of the chat
 * event stream. The agent_spec originally called for a `CustomEvent "fleet-status"`,
 * but the template's LangGraph adapter only emits TextMessage/ToolCall/Reasoning
 * events, and the one component that can emit CustomEvent (the step adaptor) is
 * switched off for custom agents. Tool results are the supported path, and they
 * carry exactly the payload the panel needs.
 *
 * Consequence worth knowing: the panel is only as fresh as the model's last call
 * to get_fleet_status. The system prompt mandates calling it first on every
 * invocation, so in practice it updates every turn.
 */

const TOOL = 'get_fleet_status';

export interface RigStatus {
  rig: string;
  verdict: 'nominal' | 'kick' | 'pump_trip' | 'deviation';
  severity: string;
  reason: string;
  flagged_channels: string[];
  step: number;
  playhead: number;
  acknowledged?: boolean;
  errors?: string[];
}

interface FleetPayload {
  fleet: RigStatus[];
  max_step: number;
}

function latestFleetResult(events: ChatStateEvent[]): FleetPayload | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (!isMessageStateEvent(ev)) continue;
    const parts = (ev.value as MessageResponse).content?.parts;
    if (!parts?.length) continue;
    for (const part of parts) {
      if (!isToolInvocationPart(part)) continue;
      const inv: ToolInvocationData | undefined = part.toolInvocation;
      if (inv?.toolName !== TOOL || inv.state !== 'result' || !inv.result) continue;
      try {
        const parsed = JSON.parse(inv.result) as FleetPayload;
        if (Array.isArray(parsed?.fleet)) return parsed;
      } catch {
        // A tool that errored returns a non-JSON string; keep looking further back.
      }
    }
  }
  return null;
}

const SEVERITY_DOT: Record<string, string> = {
  critical: 'bg-red-500',
  warning: 'bg-amber-500',
  nominal: 'bg-emerald-500',
};

export const FleetPanel: React.FC<{ events: ChatStateEvent[] }> = ({ events }) => {
  const payload = latestFleetResult(events);

  if (!payload) {
    return (
      <div className="border-b px-4 py-3 text-sm text-muted-foreground">
        Fleet — no status yet. Ask &ldquo;what&rsquo;s the fleet doing?&rdquo;
      </div>
    );
  }

  const { fleet, max_step: maxStep } = payload;
  const step = fleet[0]?.step ?? 0;

  return (
    <div className="border-b px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold">Fleet</span>
        <span className="text-xs text-muted-foreground">
          step {step} / {maxStep} · playhead {fleet[0]?.playhead ?? '—'}
        </span>
      </div>
      <div className="space-y-1">
        {fleet.map(rig => (
          <div key={rig.rig} className="flex items-center gap-2 text-sm">
            <span
              className={`inline-block size-2 rounded-full ${
                SEVERITY_DOT[rig.severity] ?? 'bg-slate-400'
              }`}
            />
            <span className="font-mono">{rig.rig}</span>
            <span className={rig.verdict === 'nominal' ? '' : 'font-semibold uppercase'}>
              {rig.verdict}
            </span>
            {rig.severity !== 'nominal' && (
              <span className="text-xs text-muted-foreground">{rig.severity}</span>
            )}
            {rig.flagged_channels.length > 0 && (
              <span className="ml-auto font-mono text-xs text-muted-foreground">
                {rig.flagged_channels.join(' · ')}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
