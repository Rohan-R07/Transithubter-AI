import { useState, useCallback, useRef, useEffect } from 'react';
import { SectionCard, StatusBadge } from '../ui/UIComponents';
import { Radio, AlertTriangle, RotateCcw, Navigation } from 'lucide-react';

// ============================================================
// NODE COLORS & TYPES
// ============================================================
const NODE_STYLES = {
  earth:   { fill: '#10b981', stroke: '#34d399', label: 'Earth' },
  relay:   { fill: '#8b5cf6', stroke: '#a78bfa', label: 'Relay' },
  target:  { fill: '#00d4ff', stroke: '#67e8f9', label: 'Target' },
  default: { fill: '#4a6a87', stroke: '#7ba3c4', label: 'Node' },
};

function getNodeStyle(type) {
  return NODE_STYLES[type] || NODE_STYLES.default;
}

// ============================================================
// EDGE STATE COLORS
// ============================================================
const EDGE_COLORS = {
  active:    '#1e3a5a',
  route:     '#00d4ff',
  disrupted: '#ef4444',
  selected:  '#f59e0b',
};

// ============================================================
// FORCE-LAYOUT HELPERS (simple static layout)
// ============================================================
function buildDefaultTopology() {
  return {
    nodes: [
      { id: 'earth',  type: 'earth',  label: 'Earth',    x: 80,  y: 220 },
      { id: 'relay1', type: 'relay',  label: 'Relay-1',  x: 250, y: 100 },
      { id: 'relay2', type: 'relay',  label: 'Relay-2',  x: 260, y: 330 },
      { id: 'relay3', type: 'relay',  label: 'Relay-3',  x: 430, y: 200 },
      { id: 'relay4', type: 'relay',  label: 'Relay-4',  x: 440, y: 380 },
      { id: 'target', type: 'target', label: 'Target',   x: 620, y: 220 },
    ],
    edges: [
      { id: 'e1',  source: 'earth',  target: 'relay1', cost: 12 },
      { id: 'e2',  source: 'earth',  target: 'relay2', cost: 10 },
      { id: 'e3',  source: 'relay1', target: 'relay3', cost: 8  },
      { id: 'e4',  source: 'relay2', target: 'relay3', cost: 14 },
      { id: 'e5',  source: 'relay2', target: 'relay4', cost: 9  },
      { id: 'e6',  source: 'relay3', target: 'target', cost: 7  },
      { id: 'e7',  source: 'relay4', target: 'target', cost: 11 },
      { id: 'e8',  source: 'relay1', target: 'relay2', cost: 6  },
      { id: 'e9',  source: 'relay3', target: 'relay4', cost: 5  },
    ],
  };
}

// ============================================================
// SPACE NODE POSITIONS & DISPLAY MAPPING
// ============================================================
export const SPACE_NODE_POSITIONS = {
  // Backend space topology IDs
  earth_ground_station:  { x: 90,  y: 230, type: 'earth',  label: 'Earth Station' },
  orbital_relay_a:       { x: 260, y: 120, type: 'relay',  label: 'Orbital Relay A' },
  orbital_relay_b:       { x: 260, y: 340, type: 'relay',  label: 'Orbital Relay B' },
  deep_space_relay:      { x: 440, y: 230, type: 'relay',  label: 'Deep Space Relay' },
  exoplanet_observatory: { x: 610, y: 230, type: 'target', label: 'Observatory' },

  // Shorthand and demo node aliases
  earth:                 { x: 90,  y: 230, type: 'earth',  label: 'Earth' },
  relay1:                { x: 260, y: 120, type: 'relay',  label: 'Relay-1' },
  relay2:                { x: 260, y: 340, type: 'relay',  label: 'Relay-2' },
  relay3:                { x: 440, y: 180, type: 'relay',  label: 'Relay-3' },
  relay4:                { x: 440, y: 350, type: 'relay',  label: 'Relay-4' },
  target:                { x: 610, y: 230, type: 'target', label: 'Target' },
};

/**
 * Map backend topology to internal format.
 * Expects backend to return { nodes: [...], links: [...] }
 */
export function mapBackendTopology(data, svgWidth = 700, svgHeight = 460) {
  if (!data || !data.nodes || (!data.links && !data.edges)) return buildDefaultTopology();

  const rawLinks = data.links || data.edges || [];

  const nodes = data.nodes.map((n, i) => {
    const predefined = SPACE_NODE_POSITIONS[n.id];
    if (predefined) {
      return {
        id: n.id,
        type: predefined.type,
        label: n.name || predefined.label,
        x: predefined.x,
        y: predefined.y,
        description: n.description,
      };
    }

    const angle = (i / data.nodes.length) * Math.PI * 2;
    const cx = svgWidth / 2, cy = svgHeight / 2, r = Math.min(svgWidth, svgHeight) * 0.35;
    return {
      id: n.id,
      type: n.node_type === 'ground_station' ? 'earth' : n.node_type === 'observatory' ? 'target' : 'relay',
      label: n.name || n.label || n.id,
      x: n.x ?? Math.round(cx + r * Math.cos(angle - Math.PI / 2)),
      y: n.y ?? Math.round(cy + r * Math.sin(angle - Math.PI / 2)),
      description: n.description,
    };
  });

  const edges = rawLinks.map((e, i) => {
    const latency = e.latency_ms ?? e.cost ?? e.latency ?? 1;
    return {
      id: e.id || `edge-${i}`,
      source: e.source,
      target: e.target || e.destination,
      cost: `${latency} ms`,
      latency_ms: latency,
      distance_km: e.distance_km,
      available: e.available ?? true,
      status: e.status || 'operational',
    };
  });

  return { nodes, edges };
}


// ============================================================
// NETWORK GRAPH COMPONENT
// ============================================================
export function NetworkGraph({
  topology,
  disabledLinks = [],
  routeEdges = [],
  selectedEdgeId,
  onEdgeSelect,
  onNodeClick,
}) {
  const { nodes, edges } = topology;

  // Build a lookup map for node positions
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  function getEdgeState(edge) {
    if (disabledLinks.includes(edge.id)) return 'disrupted';
    if (routeEdges.includes(edge.id)) return 'route';
    if (edge.id === selectedEdgeId) return 'selected';
    return 'active';
  }

  function getMidpoint(n1, n2) {
    return { x: (n1.x + n2.x) / 2, y: (n1.y + n2.y) / 2 };
  }

  return (
    <svg
      className="network-svg"
      viewBox="0 0 700 460"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block', width: '100%', height: '100%' }}
    >
      <defs>
        {/* Glow filter for route edges */}
        <filter id="glow-cyan">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-node">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        {/* Arrow marker for route */}
        <marker id="arrow-route" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill={EDGE_COLORS.route} />
        </marker>
        <marker id="arrow-active" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill={EDGE_COLORS.active} opacity="0.5" />
        </marker>

        {/* Radial gradient for nodes */}
        <radialGradient id="bg-grad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#0a1628" />
          <stop offset="100%" stopColor="#050d1a" />
        </radialGradient>
      </defs>

      {/* Background */}
      <rect width="700" height="460" fill="url(#bg-grad)" rx="12" />

      {/* Grid dots */}
      {Array.from({ length: 14 }, (_, i) =>
        Array.from({ length: 10 }, (_, j) => (
          <circle key={`${i}-${j}`} cx={i * 50 + 10} cy={j * 50 + 10} r="1" fill="rgba(255,255,255,0.04)" />
        ))
      )}

      {/* Edges */}
      {edges.map(edge => {
        const n1 = nodeMap[edge.source];
        const n2 = nodeMap[edge.target];
        if (!n1 || !n2) return null;

        const state = getEdgeState(edge);
        const isDisrupted = state === 'disrupted';
        const isRoute = state === 'route';
        const isSelected = state === 'selected';
        const mid = getMidpoint(n1, n2);

        const color = EDGE_COLORS[state];
        const strokeWidth = isRoute ? 2.5 : isSelected ? 2 : 1.5;
        const opacity = isDisrupted ? 0.4 : isRoute ? 1 : isSelected ? 0.9 : 0.35;
        const dash = isDisrupted ? '6 4' : undefined;

        return (
          <g
            key={edge.id}
            className="network-edge"
            onClick={() => onEdgeSelect?.(edge.id)}
            style={{ cursor: 'pointer' }}
          >
            {/* Clickable hit area */}
            <line
              x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
              stroke="transparent"
              strokeWidth={16}
            />
            {/* Visible line */}
            <line
              x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
              stroke={color}
              strokeWidth={strokeWidth}
              opacity={opacity}
              strokeDasharray={dash}
              filter={isRoute ? 'url(#glow-cyan)' : undefined}
              markerEnd={isRoute ? 'url(#arrow-route)' : undefined}
            />

            {/* Cost label */}
            <text
              x={mid.x} y={mid.y - 5}
              textAnchor="middle"
              fontSize="10"
              fontFamily="var(--font-mono)"
              fill={isRoute ? EDGE_COLORS.route : isDisrupted ? EDGE_COLORS.disrupted : 'rgba(123,163,196,0.6)'}
              opacity={isRoute || isDisrupted ? 1 : 0.7}
            >
              {isDisrupted ? '✕' : edge.cost}
            </text>

            {/* Selection ring on selected edge midpoint */}
            {isSelected && (
              <circle cx={mid.x} cy={mid.y} r="10" fill="none"
                stroke={EDGE_COLORS.selected} strokeWidth="1.5" opacity="0.6"
                strokeDasharray="3 3"
              />
            )}
          </g>
        );
      })}

      {/* Nodes */}
      {nodes.map(node => {
        const style = getNodeStyle(node.type);
        const isOnRoute = routeEdges.some(eid => {
          const edge = edges.find(e => e.id === eid);
          return edge && (edge.source === node.id || edge.target === node.id);
        });

        return (
          <g
            key={node.id}
            className="network-node"
            onClick={() => onNodeClick?.(node.id)}
            style={{ cursor: 'pointer' }}
          >
            {/* Outer glow ring (only on route nodes) */}
            {isOnRoute && (
              <circle
                cx={node.x} cy={node.y} r={24}
                fill="none"
                stroke={style.stroke}
                strokeWidth="1"
                opacity="0.4"
              />
            )}

            {/* Node circle */}
            <circle
              cx={node.x} cy={node.y} r={18}
              fill={style.fill}
              fillOpacity="0.18"
              stroke={style.stroke}
              strokeWidth={isOnRoute ? 2.5 : 1.5}
              filter={isOnRoute ? 'url(#glow-node)' : undefined}
            />
            <circle
              cx={node.x} cy={node.y} r={8}
              fill={style.fill}
              opacity="0.9"
            />

            {/* Node label */}
            <text
              x={node.x} y={node.y + 30}
              textAnchor="middle"
              fontSize="11"
              fontFamily="var(--font-mono)"
              fill="var(--color-text-secondary)"
              fontWeight="500"
            >
              {node.label}
            </text>

            {/* Type icon text */}
            <text
              x={node.x} y={node.y + 4}
              textAnchor="middle"
              fontSize="8"
              fontFamily="var(--font-sans)"
              fill="white"
              fontWeight="600"
            >
              {node.type === 'earth' ? '🌍' : node.type === 'target' ? '🎯' : '⬡'}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <g transform="translate(14, 390)">
        {[
          { color: EDGE_COLORS.active, label: 'Active', dash: undefined },
          { color: EDGE_COLORS.route, label: 'Route', dash: undefined },
          { color: EDGE_COLORS.disrupted, label: 'Disrupted', dash: '5 3' },
          { color: EDGE_COLORS.selected, label: 'Selected', dash: undefined },
        ].map((item, i) => (
          <g key={item.label} transform={`translate(${i * 100}, 0)`}>
            <line x1="0" y1="7" x2="20" y2="7"
              stroke={item.color} strokeWidth="2"
              strokeDasharray={item.dash}
              opacity={item.color === EDGE_COLORS.active ? 0.5 : 1}
            />
            <text x="25" y="11" fontSize="10" fontFamily="var(--font-mono)"
              fill="var(--color-text-muted)">
              {item.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// ============================================================
// LINK STATUS PANEL
// ============================================================
export function LinkStatus({ edges, disabledLinks, selectedEdgeId, onEdgeSelect, onFailLink, onRestoreLink }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {edges.map(edge => {
        const isDisabled = disabledLinks.includes(edge.id);
        const isSelected = edge.id === selectedEdgeId;
        return (
          <div
            key={edge.id}
            onClick={() => onEdgeSelect?.(edge.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '8px 12px',
              borderRadius: 8,
              border: `1px solid ${isSelected ? 'rgba(245,158,11,0.4)' : isDisabled ? 'rgba(239,68,68,0.2)' : 'var(--color-border-muted)'}`,
              background: isSelected ? 'var(--color-amber-muted)' : isDisabled ? 'var(--color-red-muted)' : 'transparent',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)',
            }}
          >
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: isDisabled ? 'var(--color-red)' : 'var(--color-green)',
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', flex: 1 }}>
              {edge.source} → {edge.target}
            </span>
            <span style={{ fontSize: 11, color: 'var(--color-text-dim)', fontFamily: 'var(--font-mono)' }}>
              {edge.cost}
            </span>
            {!isDisabled && (
              <button
                className="btn btn-danger btn-sm"
                style={{ padding: '3px 8px', fontSize: 11, marginLeft: 4 }}
                title="Simulate space signal disruption"
                onClick={(e) => { e.stopPropagation(); onFailLink?.(edge.id); }}
              >
                Fail
              </button>
            )}
            {isDisabled && (
              <button
                className="btn btn-secondary btn-sm"
                style={{ padding: '3px 8px', fontSize: 11, marginLeft: 4 }}
                title="Restore communication link"
                onClick={(e) => { e.stopPropagation(); onRestoreLink?.(edge.id); }}
              >
                Restore
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

