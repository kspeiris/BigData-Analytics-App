import React from 'react';

const iconProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': 'true',
};

export function UploadIcon() {
  return (
    <svg {...iconProps}>
      <path d="M12 16V5" />
      <path d="m8 9 4-4 4 4" />
      <path d="M20 16.5v1a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-1" />
    </svg>
  );
}

export function OverviewIcon() {
  return (
    <svg {...iconProps}>
      <rect x="4" y="4" width="7" height="7" rx="1.5" />
      <rect x="13" y="4" width="7" height="4" rx="1.5" />
      <rect x="13" y="10" width="7" height="10" rx="1.5" />
      <rect x="4" y="13" width="7" height="7" rx="1.5" />
    </svg>
  );
}

export function AnalysisIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 19h16" />
      <path d="M7 16V9" />
      <path d="M12 16V5" />
      <path d="M17 16v-4" />
    </svg>
  );
}

export function ChartsIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="m7 14 3-3 3 2 4-5" />
      <circle cx="7" cy="14" r="1" fill="currentColor" stroke="none" />
      <circle cx="10" cy="11" r="1" fill="currentColor" stroke="none" />
      <circle cx="13" cy="13" r="1" fill="currentColor" stroke="none" />
      <circle cx="17" cy="8" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function FileIcon() {
  return (
    <svg {...iconProps}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6" />
      <path d="M9 17h4" />
    </svg>
  );
}

export function DropIcon() {
  return (
    <svg {...iconProps}>
      <path d="m12 4-4 4" />
      <path d="m12 4 4 4" />
      <path d="M12 4v10" />
      <path d="M5 18a3 3 0 0 1 3-3h8a3 3 0 1 1 0 6H8a3 3 0 0 1-3-3Z" />
    </svg>
  );
}

export function BasicAnalysisIcon() {
  return (
    <svg {...iconProps}>
      <path d="M5 5h14v14H5z" />
      <path d="M5 10h14" />
      <path d="M10 5v14" />
    </svg>
  );
}

export function CorrelationIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="7" cy="8" r="2.25" />
      <circle cx="17" cy="8" r="2.25" />
      <circle cx="12" cy="17" r="2.25" />
      <path d="m8.8 9.4 1.8 5.2" />
      <path d="m15.2 9.4-1.8 5.2" />
      <path d="M9.3 8h5.4" />
    </svg>
  );
}

export function ScatterIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <circle cx="8" cy="14" r="1.4" />
      <circle cx="11.5" cy="10" r="1.4" />
      <circle cx="15" cy="12.5" r="1.4" />
      <circle cx="17.5" cy="8" r="1.4" />
    </svg>
  );
}

export function ClusteringIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="8" cy="8" r="2" />
      <circle cx="15.5" cy="7" r="2" />
      <circle cx="10" cy="15.5" r="2" />
      <circle cx="17" cy="15.5" r="2" />
      <path d="M9.5 9.5 12 13" />
      <path d="M14 8.5 11.5 14" />
      <path d="M12 15.5h3" />
    </svg>
  );
}

export function InspectorIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="11" cy="11" r="5.5" />
      <path d="m20 20-4.2-4.2" />
      <path d="M11 8v3l2 2" />
    </svg>
  );
}

export function TrendIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="m7 14 4-4 3 2 4-5" />
      <path d="m18 7 0.5 0.5" />
    </svg>
  );
}

export function OutlierIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="8" cy="10" r="1.5" />
      <circle cx="12" cy="13" r="1.5" />
      <circle cx="16" cy="9" r="1.5" />
      <circle cx="18.5" cy="5.5" r="1.5" />
      <path d="M4 19h16" />
      <path d="M4 19V5" />
    </svg>
  );
}

export function PcaIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="m7 15 4-5 3 3 4-6" />
      <path d="M7 8v7" />
      <path d="M11 10v4" />
      <path d="M14 7v6" />
      <path d="M18 5v8" />
    </svg>
  );
}
