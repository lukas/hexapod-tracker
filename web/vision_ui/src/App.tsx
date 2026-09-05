import { useEffect, useMemo, useRef, useState } from 'react'
import type {
  CalibrationJoint,
  CalibrationReport,
  Detection,
  FootTip,
  JointState,
  VisionState,
  ZeroSurveyState,
  ZeroSurveyTag,
} from './types'
import SurveyScene from './SurveyScene'

const POLL_MS = 350

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    cache: 'no-store',
    headers: {'Content-Type': 'application/json'},
    ...init,
  })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`)
  return body as T
}

function formatAngle(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}°`
}

function classForVerdict(verdict?: string) {
  return verdict === 'safe' ? 'good' : verdict === 'unsafe' ? 'bad' : 'warn'
}

function CoveragePill({label, value, total}: {label: string; value: number; total: number}) {
  const complete = value >= total
  return (
    <div className={`coverage-pill ${complete ? 'complete' : ''}`}>
      <span>{label}</span>
      <b>{value}/{total}</b>
    </div>
  )
}

function VideoOverlay({
  state,
  showTags,
  showFeet,
  showLabels,
}: {
  state: VisionState
  showTags: boolean
  showFeet: boolean
  showLabels: boolean
}) {
  const size = state.pose.image_size_px
  if (!size) return null
  const [width, height] = size
  return (
    <svg
      className="vision-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      {showTags && state.pose.tags.map((tag: Detection) => {
        const points = tag.corners_px.map((point) => point.join(',')).join(' ')
        const inferred = tag.source !== 'detected'
        return (
          <g key={tag.tag_id} className={inferred ? 'tag inferred' : 'tag'}>
            <polygon points={points} />
            <circle cx={tag.center_px[0]} cy={tag.center_px[1]} r={6} />
            {showLabels && (
              <g className="marker-label">
                <rect x={tag.center_px[0] + 9} y={tag.center_px[1] - 18} width={42} height={28} rx={7} />
                <text x={tag.center_px[0] + 30} y={tag.center_px[1] + 2}>#{tag.tag_id}</text>
              </g>
            )}
          </g>
        )
      })}
      {showFeet && state.pose.feet.map((foot: FootTip) => {
        const inferred = foot.source !== 'color'
        return (
          <g key={foot.leg} className={inferred ? 'foot inferred' : 'foot'}>
            <circle cx={foot.point_px[0]} cy={foot.point_px[1]} r={13} />
            <line x1={foot.point_px[0] - 19} y1={foot.point_px[1]} x2={foot.point_px[0] + 19} y2={foot.point_px[1]} />
            <line x1={foot.point_px[0]} y1={foot.point_px[1] - 19} x2={foot.point_px[0]} y2={foot.point_px[1] + 19} />
            {showLabels && <text x={foot.point_px[0] + 20} y={foot.point_px[1] - 14}>L{foot.leg}</text>}
          </g>
        )
      })}
    </svg>
  )
}

function JointCell({joint}: {joint?: JointState}) {
  if (!joint) return <div className="joint-cell missing">not observed</div>
  const kneeVisionUnavailable = joint.joint.endsWith('_knee')
  const delta = kneeVisionUnavailable ? undefined : joint.visual_minus_encoder_deg
  const deltaClass = delta === undefined ? '' : Math.abs(delta) <= 6 ? 'delta-ok' : 'delta-bad'
  return (
    <div className="joint-cell">
      <div className="joint-main">
        <b>{formatAngle(joint.value_deg)}</b>
        <span>{joint.source.replaceAll('_', ' ')}</span>
      </div>
      <div className={`joint-delta ${deltaClass}`}>
        {kneeVisionUnavailable ? 'encoder only · no visual knee' : delta === undefined ? 'no visual comparison' : `vision − encoder ${formatAngle(delta)}`}
      </div>
    </div>
  )
}

function LegMatrix({joints}: {joints: JointState[]}) {
  const byName = useMemo(() => new Map(joints.map((joint) => [joint.joint, joint])), [joints])
  return (
    <section className="panel leg-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">Live pose</div>
          <h2>Six-leg joint check</h2>
        </div>
        <div className="legend"><i /> within 6° <i className="red" /> investigate</div>
      </div>
      <div className="joint-grid">
        <div className="grid-head">Leg</div>
        <div className="grid-head">Yaw</div>
        <div className="grid-head">Hip</div>
        <div className="grid-head">Knee</div>
        {Array.from({length: 6}, (_, leg) => (
          <div className="joint-row" key={leg}>
            <div className="leg-name">L{leg}</div>
            <JointCell joint={byName.get(`L${leg}_yaw`)} />
            <JointCell joint={byName.get(`L${leg}_hip`)} />
            <JointCell joint={byName.get(`L${leg}_knee`)} />
          </div>
        ))}
      </div>
    </section>
  )
}

function ReportTable({
  report,
  busy,
  onApply,
}: {
  report: CalibrationReport
  busy: boolean
  onApply: () => void
}) {
  const signed = report.joints.filter((joint: CalibrationJoint) => joint.signed)
  return (
    <section className="panel report-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">Latest capture</div>
          <h2>Visual calibration report</h2>
        </div>
        <span className={`quality ${report.quality}`}>{report.quality}</span>
      </div>
      <div className="report-summary">
        <b>{report.good_signed_joint_count}/{report.signed_joint_count}</b>
        <span>signed lid joints stable across {report.sample_count} frames</span>
        <a href="/api/vision/calibration/report" download="visual-calibration.json">Download JSON</a>
      </div>
      <div className="report-table">
        {signed.map((joint) => (
          <div className="report-row" key={joint.joint}>
            <b>{joint.joint.replace('_', ' ')}</b>
            <span className={Math.abs(joint.visual_minus_encoder_deg || 0) > 6 ? 'text-bad' : ''}>
              {formatAngle(joint.visual_minus_encoder_deg)}
            </span>
            <small>MAD {formatAngle(joint.median_absolute_deviation_deg, 2)}</small>
          </div>
        ))}
      </div>
      <button
        className="primary full"
        disabled={busy || report.configuration_changed || report.good_signed_joint_count !== 12}
        onClick={onApply}
      >
        {report.configuration_changed ? 'Visual calibration applied' : 'Apply visual calibration'}
      </button>
      <p className="safety-note">Applying changes only the vision bias config. It never moves the robot or changes servo zeros.</p>
    </section>
  )
}

function stateCopy(state: string) {
  if (state === 'measured') return 'Recorded'
  if (state === 'seen_needs_another_view') return 'Another view'
  return 'Find it'
}

type LooseRecord = Record<string, unknown>

function objectValue(value: unknown): LooseRecord | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as LooseRecord
    : null
}

function numberValue(record: LooseRecord | null, names: string[]) {
  for (const name of names) {
    const value = Number(record?.[name])
    if (Number.isFinite(value)) return value
  }
  return null
}

function record3dPose(metadata: LooseRecord): number[] | null {
  const sources = [
    metadata,
    objectValue(metadata.frame),
    objectValue(metadata.camera),
  ].filter(Boolean) as LooseRecord[]
  const candidates = sources.flatMap((source) => [
    source.camera_pose_xyzw_xyz,
    source.cameraPose,
    source.camera_pose,
    source.pose,
    source.extrinsics,
  ])
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.length >= 7) {
      const pose = candidate.slice(0, 7).map(Number)
      if (pose.every(Number.isFinite)) return pose
    }
    const item = objectValue(candidate)
    if (!item) continue
    const quaternion = objectValue(item.quaternion)
    const translation = objectValue(item.translation)
    const quaternionArray = Array.isArray(item.quaternion) ? item.quaternion.map(Number) : []
    const translationArray = Array.isArray(item.translation ?? item.position)
      ? (item.translation ?? item.position) as unknown[]
      : []
    if (quaternionArray.length >= 4 && translationArray.length >= 3) {
      const pose = [...quaternionArray.slice(0, 4), ...translationArray.slice(0, 3).map(Number)]
      if (pose.every(Number.isFinite)) return pose
    }
    const pose = [
      numberValue(item, ['qx', 'x']) ?? numberValue(quaternion, ['x', 'qx']),
      numberValue(item, ['qy', 'y']) ?? numberValue(quaternion, ['y', 'qy']),
      numberValue(item, ['qz', 'z']) ?? numberValue(quaternion, ['z', 'qz']),
      numberValue(item, ['qw', 'w']) ?? numberValue(quaternion, ['w', 'qw']),
      numberValue(item, ['tx']) ?? numberValue(translation, ['x', 'tx']),
      numberValue(item, ['ty']) ?? numberValue(translation, ['y', 'ty']),
      numberValue(item, ['tz']) ?? numberValue(translation, ['z', 'tz']),
    ]
    if (pose.every((value) => value !== null && Number.isFinite(value))) return pose as number[]
  }
  return null
}

function record3dMatrix(metadata: LooseRecord, staticMetadata: LooseRecord | null, width: number, height: number): number[][] | null {
  const sources = [
    metadata,
    objectValue(metadata.frame),
    objectValue(metadata.camera),
    staticMetadata,
  ].filter(Boolean) as LooseRecord[]
  for (const source of sources) {
    const matrixCandidate = source.camera_matrix ?? source.intrinsicMatrix ?? source.K
    const flat = Array.isArray(matrixCandidate)
      ? (Array.isArray(matrixCandidate[0]) ? (matrixCandidate as unknown[][]).flat() : matrixCandidate).map(Number)
      : []
    if (flat.length === 9 && flat.every(Number.isFinite)) {
      const sourceWidth = numberValue(source, ['w', 'width', 'rgbWidth']) || width
      const sourceHeight = numberValue(source, ['h', 'height', 'rgbHeight']) || height
      // Record3D's K metadata is column-major; accept an already nested matrix too.
      const nested = Array.isArray(matrixCandidate) && Array.isArray(matrixCandidate[0])
      const rowMajor = nested
        ? flat
        : [flat[0], flat[3], flat[6], flat[1], flat[4], flat[7], flat[2], flat[5], flat[8]]
      const sx = width / sourceWidth
      const sy = height / sourceHeight
      return [
        [rowMajor[0] * sx, rowMajor[1], rowMajor[2] * sx],
        [rowMajor[3], rowMajor[4] * sy, rowMajor[5] * sy],
        [rowMajor[6], rowMajor[7], rowMajor[8]],
      ]
    }
    const rawCoeffs = source.intrinsicCoeffs
      ?? source.intrinsicMatrixCoeffs
      ?? source.intrinsics
      ?? source.perFrameIntrinsicCoeffs
    const coeffArray = Array.isArray(rawCoeffs) ? rawCoeffs.map(Number) : []
    const coeffs = objectValue(rawCoeffs)
    const fx = numberValue(coeffs, ['fx']) ?? (Number.isFinite(coeffArray[0]) ? coeffArray[0] : null)
    const fy = numberValue(coeffs, ['fy']) ?? (Number.isFinite(coeffArray[1]) ? coeffArray[1] : null)
    const cx = numberValue(coeffs, ['tx', 'cx']) ?? (Number.isFinite(coeffArray[2]) ? coeffArray[2] : null)
    const cy = numberValue(coeffs, ['ty', 'cy']) ?? (Number.isFinite(coeffArray[3]) ? coeffArray[3] : null)
    if ([fx, fy, cx, cy].every((value) => value !== null)) {
      const sourceWidth = numberValue(source, ['w', 'width', 'rgbWidth']) || width
      const sourceHeight = numberValue(source, ['h', 'height', 'rgbHeight']) || height
      return [
        [(fx as number) * width / sourceWidth, 0, (cx as number) * width / sourceWidth],
        [0, (fy as number) * height / sourceHeight, (cy as number) * height / sourceHeight],
        [0, 0, 1],
      ]
    }
  }
  return null
}

function surveyStep(status?: string) {
  if (status === 'connecting') return 1
  if (status === 'locking_origin') return 2
  if (['scanning', 'finishing', 'stopping', 'connection_lost', 'failed'].includes(status || '')) return 3
  if (['complete', 'incomplete'].includes(status || '')) return 4
  return 1
}

function SurveySchematic({survey}: {survey: ZeroSurveyState}) {
  const [view, setView] = useState<'iso' | 'top'>('iso')
  const [selected, setSelected] = useState<number | null>(null)
  const tags = survey.records.filter((tag) => tag.world_from_tag?.translation_m)
  const path = survey.camera_path_m || []
  const selectedTag = tags.find((tag) => tag.tag_id === selected)
  const targetPosition = survey.progress.robot_positions.find(
    (item) => item.position === survey.guidance?.target_position,
  )
  const targetPoint = targetPosition?.expected_world_position_m || null

  if (!tags.length) {
    return (
      <div className="survey-map empty-map">
        <div className="map-heading"><div><span>Live reconstruction</span><b>3D survey map</b></div><em>Awaiting origin lock</em></div>
        <svg viewBox="0 0 760 440" role="img" aria-label="Hexapod survey schematic waiting for tag measurements">
          <defs>
            <linearGradient id="floorFade" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#edf5ff"/><stop offset="1" stopColor="#f8fbff"/></linearGradient>
          </defs>
          <path className="map-floor" d="M90 328 L380 170 L680 330 L390 426 Z" fill="url(#floorFade)" />
          {[0, 1, 2].map((row) => [0, 1, 2].map((column) => (
            <g key={`${row}-${column}`} className="ghost-floor-tag" transform={`translate(${178 + column * 164 - row * 43} ${322 - row * 56 + column * 9})`}><rect x="-16" y="-10" width="32" height="20" rx="4"/><circle r="3"/></g>
          )))}
          <g className="ghost-robot">
            <path d="M315 248 L352 219 L416 221 L452 250 L416 282 L351 280 Z" />
            {[[335,244,250,208,183,196],[338,260,246,271,184,293],[366,279,333,354,291,391],[405,280,444,354,490,392],[431,260,528,273,598,301],[430,242,525,210,598,192]].map((points, index) => <polyline key={index} points={points.join(' ')} />)}
          </g>
          <text x="380" y="80" textAnchor="middle">Your measured tags and orientations will appear here</text>
        </svg>
      </div>
    )
  }

  const projectRaw = (point: [number, number, number]) => view === 'top'
    ? [point[0], -point[1]]
    : [(point[0] - point[1]) * 0.866, -(point[2] * 1.7 + (point[0] + point[1]) * 0.34)]
  const rawPoints = [
    ...tags.map((tag) => projectRaw(tag.world_from_tag.translation_m)),
    ...path.map((point) => projectRaw(point)),
    ...(targetPoint ? [projectRaw(targetPoint)] : []),
  ]
  const xs = rawPoints.map((point) => point[0])
  const ys = rawPoints.map((point) => point[1])
  const minX = Math.min(...xs); const maxX = Math.max(...xs)
  const minY = Math.min(...ys); const maxY = Math.max(...ys)
  const scale = Math.min(620 / Math.max(0.35, maxX - minX), 330 / Math.max(0.25, maxY - minY))
  const project = (point: [number, number, number]) => {
    const raw = projectRaw(point)
    return [70 + (raw[0] - minX) * scale, 55 + (raw[1] - minY) * scale]
  }
  const tagByFrame = new Map(tags.filter((tag) => tag.robot_frame).map((tag) => [tag.robot_frame, tag]))
  const body = tagByFrame.get('body')
  const connections: Array<[ZeroSurveyTag, ZeroSurveyTag]> = []
  for (let leg = 0; leg < 6; leg += 1) {
    const hip = tagByFrame.get(`L${leg}_coxa`)
    const knee = tagByFrame.get(`L${leg}_femur`)
    if (body && hip) connections.push([body, hip])
    if (hip && knee) connections.push([hip, knee])
  }
  const pathPoints = path.map((point) => project(point).join(',')).join(' ')

  return (
    <div className="survey-map">
      <div className="map-heading">
        <div><span>Live reconstruction</span><b>3D survey map</b></div>
        <div className="map-view-toggle"><button className={view === 'iso' ? 'active' : ''} onClick={() => setView('iso')}>Isometric</button><button className={view === 'top' ? 'active' : ''} onClick={() => setView('top')}>Top</button></div>
      </div>
      <svg viewBox="0 0 760 440" role="img" aria-label="Measured robot and floor AprilTags in three dimensions">
        <defs>
          <pattern id="grid" width="34" height="34" patternUnits="userSpaceOnUse"><path d="M34 0H0V34" fill="none" stroke="#dfe9f3" strokeWidth="1"/></pattern>
          <filter id="tagShadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#17324d" floodOpacity=".16"/></filter>
        </defs>
        <rect x="18" y="18" width="724" height="404" rx="20" fill="url(#grid)" />
        {connections.map(([first, second]) => {
          const a = project(first.world_from_tag.translation_m); const b = project(second.world_from_tag.translation_m)
          return <line key={`${first.tag_id}-${second.tag_id}`} className="robot-link" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
        })}
        {path.length > 1 && <polyline className="phone-path" points={pathPoints} />}
        {targetPoint && (() => { const target = project(targetPoint); return <g className="target-beacon" transform={`translate(${target[0]} ${target[1]})`}><circle r="19"/><circle r="7"/><text x="25" y="5">Next: {targetPosition?.position}</text></g> })()}
        {tags.map((tag) => {
          const point = tag.world_from_tag.translation_m
          const center = project(point)
          const orientation = tag.tag_y_world
          const arrow = orientation ? project([
            point[0] + orientation[0] * 0.045,
            point[1] + orientation[1] * 0.045,
            point[2] + orientation[2] * 0.045,
          ]) : null
          const role = tag.role === 'robot'
            ? 'robot'
            : ['ground', 'calibration_anchor'].includes(tag.role)
              ? 'floor'
              : 'extra'
          return (
            <g key={tag.tag_id} className={`map-tag ${role} ${tag.stable ? 'stable' : 'warming'} ${selected === tag.tag_id ? 'selected' : ''} ${survey.guidance?.target_tag_id === tag.tag_id ? 'targeted' : ''}`} transform={`translate(${center[0]} ${center[1]})`} onClick={() => setSelected(tag.tag_id)}>
              {view === 'iso' && point[2] > 0.015 && <line className="height-stem" x1="0" y1="0" x2="0" y2={Math.min(80, point[2] * scale * 1.4)} />}
              {arrow && <line className="orientation-arrow" x1="0" y1="0" x2={arrow[0] - center[0]} y2={arrow[1] - center[1]} />}
              <rect x="-12" y="-9" width="24" height="18" rx="4" filter="url(#tagShadow)" />
              <circle r="3" />
              <text x="16" y="-10">#{tag.tag_id}</text>
              <title>{`${tag.label || role} · ${tag.observations} observations`}</title>
            </g>
          )
        })}
        {survey.camera_position_m && (() => { const camera = project(survey.camera_position_m); return <g className="phone-marker" transform={`translate(${camera[0]} ${camera[1]})`}><path d="M0 -10 L8 9 L0 6 L-8 9 Z"/><text x="13" y="5">iPhone</text></g> })()}
        <g className="map-axis" transform="translate(54 376)"><line x2="42"/><line y2="-42"/><text x="47" y="4">x</text><text x="-4" y="-49">z</text></g>
      </svg>
      <div className="map-legend"><span><i className="robot" />Robot mount</span><span><i className="floor" />Floor tag</span><span><i className="extra" />Discovered extra</span><span className="legend-arrow">↗ tag +Y orientation</span></div>
      {selectedTag && <div className="selected-tag"><b>Tag #{selectedTag.tag_id}</b><span>{selectedTag.label || selectedTag.role}</span><span>{selectedTag.observations} views · {selectedTag.translation_spread_mm?.toFixed(1) || '—'} mm spread · {selectedTag.rotation_spread_deg?.toFixed(1) || '—'}°</span></div>}
    </div>
  )
}

function ZeroSurveyWorkspace({
  state,
  error,
  busy,
  floorIds,
  setFloorIds,
  originId,
  setOriginId,
  l0Id,
  setL0Id,
  bodyAnchorConfirmed,
  setBodyAnchorConfirmed,
  notice,
  connectionMode,
  setConnectionMode,
  wifiAddress,
  setWifiAddress,
  wifiStatus,
  onStart,
  onResume,
  onConnectWifi,
  onStop,
  onSave,
  onPublish,
  onPoseCheck,
}: {
  state: VisionState | null
  error: string | null
  busy: boolean
  floorIds: string
  setFloorIds: (value: string) => void
  originId: string
  setOriginId: (value: string) => void
  l0Id: string
  setL0Id: (value: string) => void
  bodyAnchorConfirmed: boolean
  setBodyAnchorConfirmed: (value: boolean) => void
  notice: string | null
  connectionMode: 'usb' | 'wifi'
  setConnectionMode: (value: 'usb' | 'wifi') => void
  wifiAddress: string
  setWifiAddress: (value: string) => void
  wifiStatus: string
  onStart: () => void
  onResume: () => void
  onConnectWifi: () => void
  onStop: () => void
  onSave: () => void
  onPublish: () => void
  onPoseCheck: () => void
}) {
  const survey = state?.zero_survey
  const active = survey?.active || false
  const measuredRobot = survey?.progress.robot_positions.filter((item) => item.state === 'measured').length || 0
  const measuredFloor = survey?.progress.ground_tag_status.filter((item) => item.state === 'measured').length || 0
  const totalRobot = survey?.progress.robot_positions.length || 37
  const totalFloor = survey?.progress.ground_tag_status.length || 7
  const provisionalTagCount = survey?.records.filter((item) => !item.stable).length || 0
  const topPositions = survey?.progress.robot_positions.filter((item) => item.kind !== 'yoke_face') || []
  const anglePositions = survey?.progress.robot_positions.filter((item) => item.kind === 'yoke_face') || []
  const topMeasured = topPositions.filter((item) => item.state === 'measured').length
  const angleMeasured = anglePositions.filter((item) => item.state === 'measured').length
  const qualityGate = survey?.progress.quality_gate
  const complete = survey?.status === 'complete'
  const interrupted = Boolean(survey?.can_resume && ['connection_lost', 'incomplete', 'failed'].includes(survey.status))
  const emptyConnectionFailure = survey?.status === 'connection_lost' && !survey.can_resume
  const step = emptyConnectionFailure ? 1 : surveyStep(survey?.status)
  const guideHeadline = survey?.status === 'connecting'
    ? 'Desktop connected — start the red Record3D stream'
    : emptyConnectionFailure
      ? 'No RGB-D frame arrived — recheck the phone'
      : survey?.guidance?.headline || survey?.instruction || 'Put the robot in zero pose and start when ready.'
  const guideDetail = survey?.status === 'connecting'
    ? 'Leave this page waiting, keep Record3D open, and stop then restart its red stream button. The first full frame will appear automatically.'
    : emptyConnectionFailure
      ? survey?.message || 'No RGB-D video reached the calibration server.'
      : survey?.guidance?.detail || survey?.message || 'The scan records tag identity, position, orientation, floor spacing, and the geometry identifiable from this pose.'
  const guideAction = survey?.status === 'connecting'
    ? 'On the phone: tap the red button to stop, then tap it again to stream.'
    : survey?.guidance?.action
  const targetPosition = survey?.guidance?.target_position
  const targetTagId = survey?.guidance?.target_tag_id
  const measuredPositionKey = (survey?.progress.robot_positions || [])
    .filter((item) => item.state === 'measured')
    .map((item) => item.position)
    .sort()
    .join('\u0000')
  const previousMeasuredPositions = useRef<Set<string> | null>(null)
  const [lastCompletedPosition, setLastCompletedPosition] = useState<string | null>(null)

  useEffect(() => {
    if (!survey) return
    const current = new Set(measuredPositionKey ? measuredPositionKey.split('\u0000') : [])
    if (previousMeasuredPositions.current !== null && active) {
      const newlyCompleted = [...current].find(
        (position) => !previousMeasuredPositions.current?.has(position),
      )
      if (newlyCompleted) setLastCompletedPosition(newlyCompleted)
    }
    previousMeasuredPositions.current = current
  }, [active, measuredPositionKey])

  return (
    <main className="calibration-app">
      <header className="studio-header">
        <div className="brand-lockup"><div className="brand-mark">HX</div><div><div className="eyebrow">Hexapod 1 · Calibration studio</div><h1>AprilTag geometry survey</h1></div></div>
        <nav className="app-nav" aria-label="Vision tools"><button className="active">Tag survey</button><button onClick={onPoseCheck}>Pose check</button></nav>
        <div className="read-only-badge"><i /> Camera only · robot stays still</div>
      </header>

      {(error || survey?.error) && <div className="error-banner"><b>Calibration needs attention:</b> {error || survey?.error}</div>}

      <div className="survey-shell">
        <aside className="wizard-rail">
          <div className="rail-intro"><span>Guided setup</span><b>About 3–5 minutes</b></div>
          {[
            ['Connect', 'Start the iPhone stream'],
            ['Set origin', 'Lock the mapped floor grid'],
            ['Walk around', 'Record 13 top + 24 side tags'],
            ['Review', 'Save and sync'],
          ].map(([title, detail], index) => {
            const number = index + 1
            return <div key={title} className={`wizard-step ${number === step ? 'current' : ''} ${number < step ? 'done' : ''}`}><span>{number < step ? '✓' : number}</span><div><b>{title}</b><small>{detail}</small></div></div>
          })}
          <div className="l0-callout"><b>L0 stays anchored by tag #{survey?.defaults.leg_zero_anchor_tag_id ?? 1}</b><span>Change this only if that particular hip tag was replaced.</span></div>
        </aside>

        <section className="survey-main">
          <div className={`guide-card status-${survey?.status || 'idle'}`}>
            <div className="guide-icon">{complete ? '✓' : active ? <span className="pulse-rings" /> : '◎'}</div>
            <div><span className="guide-kicker">Step {step} of 4 · {(survey?.status || 'ready').replaceAll('_', ' ')}</span><h2>{guideHeadline}</h2><p>{guideDetail}</p>{active && guideAction && <div className="next-action"><b>Do this now</b><span>{guideAction}</span></div>}{active && survey?.message && <small className="tracking-note">Tracking: {survey.message}</small>}</div>
            {active ? <button className="stop-survey" disabled={busy} onClick={onStop}>Stop & save partial</button> : emptyConnectionFailure ? <button className="guide-reconnect" disabled={busy} onClick={onStart}>{busy ? 'Checking…' : 'Recheck phone connection'}<span>→</span></button> : null}
          </div>

          {lastCompletedPosition && active && <div className="capture-confirmation"><b>✓ {lastCompletedPosition} complete</b><span>Enough clean frames and a genuinely different viewing angle were captured. The guide has moved to the next target.</span></div>}

          {!active && !complete && !interrupted && (
            <section className="setup-grid">
              <div className="setup-card primary-setup">
                <div className="eyebrow">Before you start</div><h2>Robot down. Tags up. Phone ready.</h2>
                <div className="transport-picker"><button className={connectionMode === 'usb' ? 'active' : ''} onClick={() => setConnectionMode('usb')}><b>USB</b><span>Best LiDAR precision</span></button><button className={connectionMode === 'wifi' ? 'active' : ''} onClick={() => setConnectionMode('wifi')}><b>Wi‑Fi</b><span>No cable · lower depth quality</span></button></div>
                {emptyConnectionFailure && <div className="connection-retry-notice"><i /><span><b>{survey.message}</b><small>{survey.instruction}</small></span></div>}
                <div className="prep-list"><span><i>1</i>Place the stationary robot in its zero pose.</span><span><i>2</i>Leave floor tags 100–105 and 112 in their mapped one-foot grid.</span><span><i>3</i>In Record3D, select {connectionMode === 'usb' ? 'USB' : 'Wi‑Fi'} under Live RGBD Video Streaming.</span><span><i>4</i>Press Connect here, then tap the red stream button on the phone. Keep Record3D open.</span></div>
                {connectionMode === 'wifi' && <div className="wifi-connect"><label>iPhone address<input value={wifiAddress} placeholder="for example 192.168.1.100 or myiPhone.local" onChange={(event) => setWifiAddress(event.target.value)} /></label><button disabled={busy || !wifiAddress.trim()} onClick={onConnectWifi}>Connect phone</button><small>{wifiStatus}. Requires Record3D 1.11+ and its Wi‑Fi Streaming extension; USB remains more accurate.</small></div>}
                <button className="launch-survey" disabled={busy || !survey?.available} onClick={onStart}>{busy ? 'Checking for RGB‑D frames…' : emptyConnectionFailure ? 'Recheck phone connection' : 'Start iPhone LiDAR calibration'}<span>→</span></button>
              </div>
              <div className="setup-card settings-card">
                <div className="panel-heading"><div><div className="eyebrow">Known setup</div><h2>Tag references</h2></div><span className="defaults-chip">editable</span></div>
                <label>Floor origin tag<input value={originId} inputMode="numeric" onChange={(event) => setOriginId(event.target.value)} /></label>
                <label>Floor tags to find<input value={floorIds} onChange={(event) => setFloorIds(event.target.value)} /></label>
                <label>L0 hip identity tag<input value={l0Id} inputMode="numeric" onChange={(event) => setL0Id(event.target.value)} /></label>
                <p>The other robot IDs may change. Calibration fills each named physical position with whichever stable tag is actually there.</p>
                <div className={`lab-preflight ${survey?.robot_lab.status === 'ready' ? 'ready' : ''}`}><i />
                  <span><b>{survey?.robot_lab.status === 'ready' ? 'Robot Lab ready' : 'Robot Lab credential needed'}</b><small>{survey?.robot_lab.status === 'ready' ? `Credential loaded from ${survey.robot_lab.credential_source || 'the server environment'}; the reviewed config will publish automatically.` : survey?.robot_lab.error || 'Use 1Password, a protected credential file, or the server environment; the scan can still run now.'}</small></span>
                </div>
              </div>
            </section>
          )}

          {!active && !complete && interrupted && survey && (
            <section className="resume-card">
              <div><div className="eyebrow">Saved checkpoint</div><h2>Your scan is intact</h2><p>{measuredRobot}/{totalRobot} robot positions and {measuredFloor}/{totalFloor} floor tags are finished; {provisionalTagCount} additional tag identities and pose seeds are retained. Reconnect the phone, re-lock the floor, then add the different angles needed to finish them.</p></div>
              <div className="resume-controls">
                <div className="transport-picker compact"><button className={connectionMode === 'usb' ? 'active' : ''} onClick={() => setConnectionMode('usb')}><b>USB</b><span>Recommended</span></button><button className={connectionMode === 'wifi' ? 'active' : ''} onClick={() => setConnectionMode('wifi')}><b>Wi‑Fi</b><span>Wireless</span></button></div>
                {connectionMode === 'wifi' && <div className="wifi-connect compact"><label>iPhone address<input value={wifiAddress} placeholder="myiPhone.local" onChange={(event) => setWifiAddress(event.target.value)} /></label><button disabled={busy || !wifiAddress.trim()} onClick={onConnectWifi}>Reconnect phone</button><small>{wifiStatus}</small></div>}
                <button className="launch-survey" disabled={busy || !survey.can_resume} onClick={onResume}>{busy ? 'Reconnecting…' : 'Continue calibration'}<span>→</span></button>
                <button className="text-action" disabled={busy} onClick={onStart}>Discard checkpoint and start over</button>
              </div>
            </section>
          )}

          {(active || complete || interrupted) && survey && (
            <>
              <div className="live-grid">
                <section className="camera-card">
                  <div className="map-heading"><div><span>Record3D RGB + LiDAR</span><b>Live positioning view</b></div><em>Full frame · no crop</em></div>
                  <div className={`survey-camera ${survey.camera_frame_available ? 'has-frame' : 'waiting'} ${active ? 'is-live' : 'is-stale'}`}>
                    {survey.camera_frame_available ? <><img src={`/api/vision/zero-survey/frame.jpg?v=${survey.camera_frame_version}`} alt={active ? 'Live full iPhone positioning frame' : 'Last saved full iPhone frame'} draggable={false} /><div className={`camera-frame-label ${active ? 'live' : 'stale'}`}>{active ? '● LIVE · FULL FRAME' : 'LAST SAVED FRAME · RECONNECT TO UPDATE'}</div>{!active && interrupted && <div className="stale-frame-overlay"><b>The phone feed is disconnected</b><span>This is the final complete frame received, not the phone’s current view.</span></div>}</> : <div className="waiting-camera"><div className="phone-glyph">▯</div><b>Desktop connected; waiting for Record3D video</b><span>If the phone says it is streaming but no frame appears, stop and restart the red {connectionMode === 'wifi' ? 'Wi‑Fi' : 'USB'} button once.</span></div>}
                  </div>
                  {active && survey.guidance?.accepted_frames !== undefined && <div className="capture-progress"><div><span>Clean frames</span><b>{Math.min(survey.guidance.accepted_frames, survey.guidance.required_frames || 0)}/{survey.guidance.required_frames}</b><i><em style={{width: `${Math.min(100, survey.guidance.accepted_frames / Math.max(1, survey.guidance.required_frames || 1) * 100)}%`}} /></i></div><div><span>Different-angle proof</span><b>{(survey.guidance.viewpoint_span_deg || 0).toFixed(0)}°/{(survey.guidance.required_viewpoint_span_deg || 0).toFixed(0)}°</b><i><em style={{width: `${Math.min(100, (survey.guidance.viewpoint_span_deg || 0) / Math.max(1, survey.guidance.required_viewpoint_span_deg || 1) * 100)}%`}} /></i></div></div>}
                  <div className="camera-stats"><span><b>{survey.frame_sequence}</b> analyzed frames</span><span><b>{survey.detected_tag_ids.length}</b> tags in last analysis</span><span><b>{survey.elapsed_s.toFixed(0)}s</b> elapsed</span></div>
                  {survey.quality && <div className={`quality-coach ${survey.quality.level}`}><div className="quality-head"><i /><span><b>{survey.quality.headline}</b><small>{survey.quality.suggestion}</small></span></div><div className="quality-metrics"><span><b>{survey.quality.reprojection_rms_px === null ? '—' : `${survey.quality.reprojection_rms_px.toFixed(2)} px`}</b>corner error</span><span><b>{survey.quality.depth_plane_rms_mm === null ? '—' : `${survey.quality.depth_plane_rms_mm.toFixed(1)} mm`}</b>LiDAR plane</span><span><b>{survey.quality.translation_spread_mm === null ? '—' : `${survey.quality.translation_spread_mm.toFixed(1)} mm`}</b>position spread</span><span><b>{survey.quality.rotation_spread_deg === null ? '—' : `${survey.quality.rotation_spread_deg.toFixed(1)}°`}</b>angle spread</span><span><b>{survey.quality.camera_speed_m_s === null ? '—' : `${survey.quality.camera_speed_m_s.toFixed(2)} m/s`}</b>phone speed</span></div></div>}
                </section>
                <SurveyScene survey={survey} />
              </div>

              <section className="coverage-board">
                <div className="coverage-heading"><div><div className="eyebrow">Position-based coverage</div><h2>{topMeasured}/{topPositions.length || 13} top + chassis · {angleMeasured}/{anglePositions.length || 24} vertical angle tags · {measuredFloor}/{totalFloor} floor</h2></div><div className="coverage-total"><span style={{width: `${((measuredRobot + measuredFloor) / Math.max(1, totalRobot + totalFloor)) * 100}%`}} /></div></div>
                {qualityGate && <div className={`geometry-gate ${qualityGate.passed ? 'passed' : 'checking'}`}><div><b>{qualityGate.passed ? 'Geometry quality passed' : 'Geometry quality checking'}</b><span>{qualityGate.passed ? 'The mapped floor residuals and all required positions meet the save limits.' : qualityGate.failing_checks[0]}</span></div><div><span><b>{qualityGate.joint_floor_reprojection_rms_px == null ? '—' : `${qualityGate.joint_floor_reprojection_rms_px.toFixed(2)} px`}</b>floor grid fit</span><span><b>{qualityGate.floor_position_rms_mm === null ? '—' : `${qualityGate.floor_position_rms_mm.toFixed(1)} mm`}</b>floor position RMS</span><span><b>{qualityGate.floor_height_rms_mm === null ? '—' : `${qualityGate.floor_height_rms_mm.toFixed(1)} mm`}</b>floor height RMS</span><span><b>{qualityGate.floor_rotation_rms_deg === null ? '—' : `${qualityGate.floor_rotation_rms_deg.toFixed(1)}°`}</b>floor angle RMS</span></div></div>}
                <div className="position-groups">
                  <div className="robot-position-sections"><div><h3>Top + chassis tags <small>{topMeasured}/{topPositions.length}</small></h3><div className="position-list">{topPositions.map((item) => <div key={item.position} className={`position-item ${item.state} ${targetPosition === item.position ? 'targeted' : ''}`}><i>{item.state === 'measured' ? '✓' : item.state === 'seen_needs_another_view' ? '↻' : '·'}</i><span><b>{item.position}</b><small>{item.tag_id === null ? 'No tag seen yet' : `tag #${item.tag_id}${item.replacement ? ' · replacement' : ''} · ${item.observations || 0} frames · ${(item.viewpoint_span_deg || 0).toFixed(0)}° span`}</small></span><em>{targetPosition === item.position ? 'Do now' : stateCopy(item.state)}</em></div>)}</div></div><div><h3>Vertical angle tags <small>{angleMeasured}/{anglePositions.length} · four per leg</small></h3><div className="position-list angle-list">{anglePositions.map((item) => <div key={item.position} className={`position-item ${item.state} ${targetPosition === item.position ? 'targeted' : ''}`}><i>{item.state === 'measured' ? '✓' : item.state === 'seen_needs_another_view' ? '↻' : '·'}</i><span><b>{item.position}</b><small>{item.tag_id === null ? 'No tag seen yet' : `tag #${item.tag_id}${item.replacement ? ' · replacement' : ''} · ${item.observations || 0} frames · ${(item.viewpoint_span_deg || 0).toFixed(0)}° span`}</small></span><em>{targetPosition === item.position ? 'Do now' : stateCopy(item.state)}</em></div>)}</div></div></div>
                  <div><h3>Floor tags</h3><div className="floor-list">{survey.progress.ground_tag_status.map((item) => <div key={item.tag_id} className={`floor-item ${item.state} ${targetTagId === item.tag_id ? 'targeted' : ''}`}><b>#{item.tag_id}</b><span>{targetTagId === item.tag_id ? 'Do now' : stateCopy(item.state)}</span><small>{item.observations} frames · {(item.viewpoint_span_deg || 0).toFixed(0)}° span</small></div>)}</div>{survey.progress.discovered_unexpected_tag_ids.length > 0 && <div className="extra-tags"><b>Also discovered</b><span>{survey.progress.discovered_unexpected_tag_ids.map((id) => `#${id}`).join(', ')}</span></div>}</div>
                </div>
              </section>
            </>
          )}

          {(complete || survey?.status === 'incomplete') && survey && (
            <section className={`review-card ${complete ? 'complete' : 'partial'}`}>
              <div className="review-heading"><div className="review-check">{complete ? '✓' : '!'}</div><div><div className="eyebrow">Survey review</div><h2>{complete ? 'Every required position is recorded' : 'A partial survey was saved'}</h2><p>{complete ? 'Review the schematic and confirm the one fixed body-frame reference before creating the robot configuration.' : survey.instruction}</p></div><a href="/api/vision/zero-survey/result" download="zero-pose-survey.json">Download measurements</a></div>
              {complete && <div className="finalize-grid"><label className="anchor-confirm"><input type="checkbox" checked={bodyAnchorConfirmed} onChange={(event) => setBodyAnchorConfirmed(event.target.checked)} /><span><b>Chassis tag #0 is still in its original mount and orientation.</b><small>This is the one fixed reference needed to learn all other tag mounts.</small></span></label><button className="save-config" disabled={busy || !bodyAnchorConfirmed || survey.reviewed_config_available} onClick={onSave}>{survey.reviewed_config_available ? 'Configuration saved' : 'Save configuration & update Robot Lab'}<span>→</span></button></div>}
              {notice && <div className="success-notice">{notice}</div>}
              {survey.robot_lab.status === 'published' && <div className="lab-sync published"><b>Robot Lab updated</b><span>Survey and calibrated tracker configuration are saved as durable artifacts.</span>{survey.robot_lab.url && <a href={survey.robot_lab.url} target="_blank" rel="noreferrer">Open result ↗</a>}</div>}
              {['failed', 'not_configured'].includes(survey.robot_lab.status) && survey.reviewed_config_available && <div className="lab-sync failed"><div><b>Robot Lab still needs this update</b><span>{survey.robot_lab.error || 'The vision server needs the Robot Lab credential.'}</span></div><button disabled={busy} onClick={onPublish}>Retry sync</button></div>}
              {survey.status === 'incomplete' && <button className="secondary restart" disabled={busy || !survey.can_resume} onClick={onResume}>Continue this calibration</button>}
            </section>
          )}
        </section>
      </div>
    </main>
  )
}

export default function App() {
  const [state, setState] = useState<VisionState | null>(null)
  const [report, setReport] = useState<CalibrationReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [cameraInput, setCameraInput] = useState('0')
  const [showTags, setShowTags] = useState(true)
  const [showFeet, setShowFeet] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [surveyGaits, setSurveyGaits] = useState<number[]>([1, 11])
  const [surveySpeed, setSurveySpeed] = useState('30')
  const [surveyDuration, setSurveyDuration] = useState('8')
  const [surveyRecoveries, setSurveyRecoveries] = useState('2')
  const [surveyAck, setSurveyAck] = useState(false)
  const [showSurvey, setShowSurvey] = useState(false)
  const [activeView, setActiveView] = useState<'survey' | 'pose'>('survey')
  const [zeroFloorIds, setZeroFloorIds] = useState('100, 101, 102, 103, 104, 105, 112')
  const [zeroOriginId, setZeroOriginId] = useState('104')
  const [zeroL0Id, setZeroL0Id] = useState('1')
  const [zeroDefaultsLoaded, setZeroDefaultsLoaded] = useState(false)
  const [bodyAnchorConfirmed, setBodyAnchorConfirmed] = useState(false)
  const [zeroNotice, setZeroNotice] = useState<string | null>(null)
  const [zeroConnectionMode, setZeroConnectionMode] = useState<'usb' | 'wifi'>('usb')
  const [zeroWifiAddress, setZeroWifiAddress] = useState('')
  const [wifiStatus, setWifiStatus] = useState('Not connected')
  const wifiPeer = useRef<RTCPeerConnection | null>(null)
  const wifiVideo = useRef<HTMLVideoElement | null>(null)
  const wifiMetadata = useRef(new Map<number, LooseRecord>())
  const wifiStaticMetadata = useRef<LooseRecord | null>(null)
  const wifiUploadEnabled = useRef(false)
  const wifiUploadPending = useRef(false)
  const wifiLastUpload = useRef(0)
  const wifiSequence = useRef(0)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const next = await api<VisionState>('/api/vision/state')
        if (!cancelled) {
          setState(next)
          setError(null)
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : String(caught))
      }
    }
    void refresh()
    const timer = window.setInterval(refresh, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!state?.calibration.report_available) return
    api<CalibrationReport>('/api/vision/calibration/report').then(setReport).catch(() => undefined)
  }, [state?.calibration.report_available, state?.calibration.status])

  useEffect(() => {
    if (state?.camera.requested_index === undefined) return
    setCameraInput(String(state.camera.requested_index))
  }, [state?.camera.requested_index])

  useEffect(() => {
    if (!state?.zero_survey?.defaults || zeroDefaultsLoaded) return
    setZeroFloorIds(state.zero_survey.defaults.floor_tag_ids.join(', '))
    setZeroOriginId(String(state.zero_survey.defaults.origin_tag_id))
    setZeroL0Id(String(state.zero_survey.defaults.leg_zero_anchor_tag_id))
    setZeroConnectionMode(state.zero_survey.connection_mode || state.zero_survey.defaults.connection_mode || 'usb')
    setZeroWifiAddress(state.zero_survey.wifi_address || state.zero_survey.defaults.wifi_address || '')
    setZeroDefaultsLoaded(true)
  }, [state?.zero_survey?.defaults, zeroDefaultsLoaded])

  useEffect(() => () => {
    wifiUploadEnabled.current = false
    wifiPeer.current?.close()
    wifiPeer.current = null
  }, [])

  useEffect(() => {
    if (state?.zero_survey && !state.zero_survey.active) {
      wifiUploadEnabled.current = false
    }
  }, [state?.zero_survey?.active])

  const connectRecord3dWifi = async () => {
    const host = zeroWifiAddress.trim().replace(/\/$/, '')
    if (!host) throw new Error('Enter the address shown by Record3D')
    const baseUrl = /^https?:\/\//i.test(host) ? host : `http://${host}`
    setWifiStatus('Connecting to Record3D…')
    wifiPeer.current?.close()
    wifiVideo.current = null
    const staticResponse = await fetch(`${baseUrl}/metadata`, {cache: 'no-store'})
    if (!staticResponse.ok) throw new Error(`Record3D metadata returned ${staticResponse.status}`)
    wifiStaticMetadata.current = await staticResponse.json() as LooseRecord
    const offerResponse = await fetch(`${baseUrl}/getOffer`, {cache: 'no-store'})
    if (!offerResponse.ok) throw new Error(offerResponse.status === 403 ? 'Another viewer is already connected to the phone' : `Record3D offer returned ${offerResponse.status}`)
    const rawOffer = await offerResponse.json() as LooseRecord
    const offer = {
      type: String(rawOffer.type || 'offer') as RTCSdpType,
      sdp: String(rawOffer.sdp || rawOffer.data || ''),
    }
    if (!offer.sdp) throw new Error('Record3D returned an offer without SDP')
    const Transform = (window as unknown as {RTCRtpScriptTransform?: new (worker: Worker, options: LooseRecord) => unknown}).RTCRtpScriptTransform
    if (!Transform) throw new Error('This browser cannot read Record3D pose metadata; use USB or a current Chromium browser')
    const peer = new RTCPeerConnection()
    wifiPeer.current = peer
    const connected = new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error('Timed out waiting for the Record3D video track')), 12000)
      peer.ontrack = (event) => {
        window.clearTimeout(timeout)
        const worker = new Worker(new URL('./record3dMetadataWorker.ts', import.meta.url), {type: 'module'})
        worker.onmessage = (message: MessageEvent<{frameTimestamp?: number; metadata?: LooseRecord; error?: string}>) => {
          if (message.data.frameTimestamp !== undefined && message.data.metadata) {
            wifiMetadata.current.set(message.data.frameTimestamp, message.data.metadata)
            if (wifiMetadata.current.size > 120) {
              const oldest = wifiMetadata.current.keys().next().value
              if (oldest !== undefined) wifiMetadata.current.delete(oldest)
            }
          } else if (message.data.error) {
            setWifiStatus(`Metadata warning: ${message.data.error}`)
          }
        }
        ;(event.receiver as unknown as {transform: unknown}).transform = new Transform(worker, {name: 'receiverTransform'})
        const video = document.createElement('video')
        video.playsInline = true
        video.autoplay = true
        video.muted = true
        video.srcObject = event.streams[0]
        wifiVideo.current = video
        void video.play()
        const canvas = document.createElement('canvas')
        const sendFrame: VideoFrameRequestCallback = (_now, frameMetadata) => {
          if (wifiPeer.current !== peer) return
          video.requestVideoFrameCallback(sendFrame)
          const rtpTimestamp = frameMetadata.rtpTimestamp
          const metadata = rtpTimestamp === undefined ? undefined : wifiMetadata.current.get(rtpTimestamp)
          if (rtpTimestamp !== undefined && metadata) wifiMetadata.current.delete(rtpTimestamp)
          if (!wifiUploadEnabled.current || !metadata || wifiUploadPending.current || performance.now() - wifiLastUpload.current < 250) return
          if (!video.videoWidth || !video.videoHeight) return
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          canvas.getContext('2d', {alpha: false})?.drawImage(video, 0, 0)
          const rgbWidth = Math.floor(video.videoWidth / 2)
          const pose = record3dPose(metadata)
          const matrix = record3dMatrix(metadata, wifiStaticMetadata.current, rgbWidth, video.videoHeight)
          if (!pose || !matrix) {
            setWifiStatus('Video connected; waiting for Record3D 1.11 pose metadata')
            return
          }
          const maxDepth = numberValue(metadata, ['maxDepth', 'maxDepthM', 'max_depth_m'])
            ?? numberValue(wifiStaticMetadata.current, ['maxDepth', 'maxDepthM', 'max_depth_m'])
            ?? 3
          wifiUploadPending.current = true
          wifiLastUpload.current = performance.now()
          wifiSequence.current += 1
          void api('/api/vision/zero-survey/wifi-frame', {
            method: 'POST',
            body: JSON.stringify({
              sequence: wifiSequence.current,
              captured_unix: Date.now() / 1000,
              rgbd_jpeg_base64: canvas.toDataURL('image/jpeg', 0.9),
              camera_matrix: matrix,
              camera_pose_xyzw_xyz: pose,
              max_depth_m: maxDepth,
            }),
          }).then(() => setWifiStatus('Streaming RGB-D, pose, and intrinsics')).catch((caught) => {
            setWifiStatus(caught instanceof Error ? caught.message : String(caught))
          }).finally(() => { wifiUploadPending.current = false })
        }
        video.requestVideoFrameCallback(sendFrame)
        resolve()
      }
    })
    peer.onconnectionstatechange = () => {
      if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) {
        wifiVideo.current = null
        setWifiStatus('Phone connection lost — restart its stream, then reconnect here')
      }
    }
    await peer.setRemoteDescription(offer)
    await peer.setLocalDescription(await peer.createAnswer())
    if (peer.iceGatheringState !== 'complete') {
      await new Promise<void>((resolve) => {
        peer.addEventListener('icegatheringstatechange', () => {
          if (peer.iceGatheringState === 'complete') resolve()
        })
      })
    }
    // Record3D's Wi-Fi signaling API calls this endpoint /sendAnswer.
    const answerResponse = await fetch(`${baseUrl}/sendAnswer`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({type: 'answer', data: peer.localDescription?.sdp}),
    })
    if (!answerResponse.ok) throw new Error(`Record3D answer returned ${answerResponse.status}`)
    await connected
    setWifiStatus('Phone connected; ready to calibrate')
  }

  const switchCamera = async (index: number) => {
    setBusy(true)
    setCameraInput(String(index))
    try {
      const next = await api<VisionState>('/api/vision/camera', {method: 'POST', body: JSON.stringify({index})})
      setState(next)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const setCameraEnabled = async (enabled: boolean) => {
    setBusy(true)
    try {
      const next = await api<VisionState>(`/api/vision/camera/${enabled ? 'start' : 'stop'}`, {
        method: 'POST',
        body: enabled ? JSON.stringify({index: Number(cameraInput)}) : '{}',
      })
      setState(next)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const rescanCameras = async () => {
    setBusy(true)
    try {
      const next = await api<VisionState>('/api/vision/cameras/rescan', {method: 'POST', body: '{}'})
      setState(next)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const startCalibration = async () => {
    setBusy(true)
    try {
      await api('/api/vision/calibration/start', {method: 'POST', body: '{}'})
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const cancelCalibration = async () => {
    await api('/api/vision/calibration/cancel', {method: 'POST', body: '{}'})
  }

  const applyCalibration = async () => {
    setBusy(true)
    try {
      await api('/api/vision/calibration/apply', {method: 'POST', body: '{}'})
      const latest = await api<CalibrationReport>('/api/vision/calibration/report')
      setReport(latest)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const toggleSurveyGait = (gait: number) => {
    setSurveyGaits((current) => (
      current.includes(gait)
        ? current.filter((value) => value !== gait)
        : [...current, gait].sort((a, b) => a - b)
    ))
  }

  const startSurvey = async () => {
    setBusy(true)
    try {
      await api('/api/vision/survey/start', {
        method: 'POST',
        body: JSON.stringify({
          acknowledge_motion: surveyAck,
          gaits: surveyGaits,
          speed_mm_s: Number(surveySpeed),
          direction_s: Number(surveyDuration),
          settle_s: 1.5,
          adaptive_centering: true,
          soft_recovery: true,
          max_recoveries: Number(surveyRecoveries),
        }),
      })
      setSurveyAck(false)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const stopSurvey = async () => {
    setBusy(true)
    try {
      await api('/api/vision/survey/stop', {method: 'POST', body: '{}'})
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const startZeroSurvey = async () => {
    setBusy(true)
    setZeroNotice(null)
    setBodyAnchorConfirmed(false)
    try {
      if (zeroConnectionMode === 'wifi' && !wifiVideo.current) {
        await connectRecord3dWifi()
      }
      const floor_tag_ids = zeroFloorIds.split(',').map((value) => Number(value.trim())).filter(Number.isFinite)
      if (!floor_tag_ids.length) throw new Error('Enter at least one floor tag ID')
      await api('/api/vision/zero-survey/start', {
        method: 'POST',
        body: JSON.stringify({
          record3d_device: state?.zero_survey.defaults.record3d_device ?? 0,
          connection_mode: zeroConnectionMode,
          wifi_address: zeroWifiAddress,
          origin_tag_id: Number(zeroOriginId),
          floor_tag_ids,
          marker_size_mm: state?.zero_survey.defaults.marker_size_mm ?? 27,
          body_anchor_tag_id: state?.zero_survey.defaults.body_anchor_tag_id ?? 0,
          leg_zero_anchor_tag_id: Number(zeroL0Id),
        }),
      })
      wifiUploadEnabled.current = zeroConnectionMode === 'wifi'
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const stopZeroSurvey = async () => {
    setBusy(true)
    try {
      await api('/api/vision/zero-survey/stop', {method: 'POST', body: '{}'})
      wifiUploadEnabled.current = false
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const resumeZeroSurvey = async () => {
    setBusy(true)
    setZeroNotice(null)
    try {
      if (zeroConnectionMode === 'wifi' && !wifiVideo.current) {
        await connectRecord3dWifi()
      }
      await api('/api/vision/zero-survey/resume', {
        method: 'POST',
        body: JSON.stringify({
          connection_mode: zeroConnectionMode,
          wifi_address: zeroWifiAddress,
          record3d_device: state?.zero_survey.defaults.record3d_device ?? 0,
        }),
      })
      wifiUploadEnabled.current = zeroConnectionMode === 'wifi'
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const prepareRecord3dWifi = async () => {
    setBusy(true)
    try {
      await connectRecord3dWifi()
      setError(null)
    } catch (caught) {
      setWifiStatus(caught instanceof Error ? caught.message : String(caught))
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const saveZeroSurvey = async () => {
    setBusy(true)
    try {
      const saved = await api<{config_path: string; robot_lab: {status: string; url?: string; error?: string}}>('/api/vision/zero-survey/save', {
        method: 'POST',
        body: JSON.stringify({confirm_body_anchor_unchanged: bodyAnchorConfirmed}),
      })
      setZeroNotice(saved.robot_lab.status === 'published'
        ? 'Configuration saved locally and published to Robot Lab.'
        : `Configuration saved locally. Robot Lab: ${saved.robot_lab.error || saved.robot_lab.status}.`)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  const publishZeroSurvey = async () => {
    setBusy(true)
    try {
      const published = await api<{status: string; url?: string; error?: string}>('/api/vision/zero-survey/publish', {method: 'POST', body: '{}'})
      setZeroNotice(published.status === 'published' ? 'Robot Lab is updated.' : published.error || published.status)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  if (activeView === 'survey') {
    return <ZeroSurveyWorkspace
      state={state}
      error={error}
      busy={busy}
      floorIds={zeroFloorIds}
      setFloorIds={setZeroFloorIds}
      originId={zeroOriginId}
      setOriginId={setZeroOriginId}
      l0Id={zeroL0Id}
      setL0Id={setZeroL0Id}
      bodyAnchorConfirmed={bodyAnchorConfirmed}
      setBodyAnchorConfirmed={setBodyAnchorConfirmed}
      notice={zeroNotice}
      connectionMode={zeroConnectionMode}
      setConnectionMode={setZeroConnectionMode}
      wifiAddress={zeroWifiAddress}
      setWifiAddress={setZeroWifiAddress}
      wifiStatus={wifiStatus}
      onStart={() => void startZeroSurvey()}
      onResume={() => void resumeZeroSurvey()}
      onConnectWifi={() => void prepareRecord3dWifi()}
      onStop={() => void stopZeroSurvey()}
      onSave={() => void saveZeroSurvey()}
      onPublish={() => void publishZeroSurvey()}
      onPoseCheck={() => setActiveView('pose')}
    />
  }

  const safety = state?.pose.safety
  const readiness = state?.readiness
  const collecting = state?.calibration.status === 'collecting'
  const survey = state?.survey
  const surveyActive = survey?.active || false
  const cameraDevices = state?.camera.devices || []
  const cameraTransitioning = state?.camera.status === 'starting' || state?.camera.status === 'switching'
  const safetyReasons = safety?.unsafe_reasons.length
    ? safety.unsafe_reasons
    : safety?.unknown_reasons || []
  const safetyMessage = !state?.camera.enabled
    ? 'Camera is off; pose safety has not been evaluated.'
    : safetyReasons[0] || safety?.warnings[0] || 'No unsafe condition is currently detected.'

  return (
    <main>
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">HX</div>
          <div>
            <div className="eyebrow">STS3215 · Vision lab</div>
            <h1>Visual calibration</h1>
          </div>
        </div>
        <nav className="app-nav compact" aria-label="Vision tools"><button onClick={() => setActiveView('survey')}>Tag survey</button><button className="active">Pose check</button></nav>
        <div className="top-status">
          <span className={surveyActive ? 'motion-dot' : 'readonly-dot'} />
          {surveyActive ? 'Guarded gait survey active' : 'Observation · survey motion gated'}
          <span className="divider" />
          <b>{state?.performance.fps.toFixed(1) || '—'} fps</b>
          <span>{state?.performance.frame_age_ms === null || state?.performance.frame_age_ms === undefined ? '—' : `${state.performance.frame_age_ms.toFixed(0)} ms`}</span>
        </div>
      </header>

      {error && <div className="error-banner"><b>Vision service:</b> {error}</div>}

      <div className="workspace">
        <section className="video-column">
          <div className="video-stage">
            {state?.camera.enabled && state.camera.status === 'running' && <img src={`/api/vision/frame.mjpg?camera=${state.camera.active_index}`} alt="Live camera feed of the hexapod" />}
            {state && state.camera.status === 'running' && <VideoOverlay state={state} showTags={showTags} showFeet={showFeet} showLabels={showLabels} />}
            {!state?.camera.enabled && surveyActive ? (
              <div className="video-empty camera-off-prompt survey-recording">
                <b>Survey recorder owns the camera</b>
                <span>Raw video is still recording. Live Vision preview returns automatically afterward.</span>
              </div>
            ) : !state?.camera.enabled ? (
              <div className="video-empty camera-off-prompt">
                <b>Camera is off</b>
                <span>Choose an input, then start the camera when you are ready.</span>
                <button className="primary" disabled={busy} onClick={() => void setCameraEnabled(true)}>Start camera</button>
              </div>
            ) : cameraTransitioning ? (
              <div className="video-empty camera-off-prompt">
                <b>{state.camera.status === 'switching' ? 'Switching camera…' : 'Opening camera…'}</b>
                <span>{cameraDevices.find((device) => device.index === state.camera.requested_index)?.name || `Camera ${state.camera.requested_index}`}</span>
              </div>
            ) : !state?.ok && <div className="video-empty">Waiting for the first camera frame…</div>}
            <div className="video-badges">
              <span>Camera {state?.camera.active_index ?? '—'}</span>
              <span>{state?.pose.pose_reference || 'no pose'} frame</span>
              <span>preview {state?.performance.image_size_px?.join(' × ') || '—'}</span>
              {state?.camera.native_luma && (
                <span className="native-badge">tags {state.performance.detection_image_size_px?.join(' × ')} luma</span>
              )}
            </div>
          </div>
          <div className="video-toolbar">
            <div className="overlay-options">
              <span>Overlay</span>
              <label><input type="checkbox" checked={showTags} onChange={(event) => setShowTags(event.target.checked)} /> Tags</label>
              <label><input type="checkbox" checked={showFeet} onChange={(event) => setShowFeet(event.target.checked)} /> Feet</label>
              <label><input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)} /> IDs</label>
            </div>
            <span className="camera-message">{state?.camera.error || 'Latest-frame streaming: stale frames are dropped.'}</span>
          </div>
          <LegMatrix joints={state?.pose.joints || []} />
          {report && <ReportTable report={report} busy={busy || surveyActive} onApply={() => void applyCalibration()} />}
        </section>

        <aside className="control-column">
          <section className={`panel safety-card ${classForVerdict(safety?.verdict)}`}>
            <div className="eyebrow">Pose safety</div>
            <div className="big-status">{safety?.verdict?.toUpperCase() || 'WAITING'}</div>
            <p>{safetyMessage}</p>
            <div className="safety-metrics">
              <span>IMU tilt <b>{formatAngle(safety?.imu_tilt_deg)}</b></span>
              <span>vision tilt <b>{formatAngle(state?.pose.body_tilt_deg)}</b></span>
              <span>reference <b>{state?.pose.pose_reference || '—'}</b></span>
            </div>
          </section>

          <section className={`panel survey-card ${surveyActive ? 'is-active' : ''} ${showSurvey ? 'is-open' : ''}`}>
            <div className="panel-heading compact">
              <div>
                <div className="eyebrow">Recorded experiment</div>
                <h2>Gait survey</h2>
              </div>
              <span className={`survey-state ${survey?.status || 'idle'}`}>
                {survey?.status || 'idle'}
              </span>
            </div>

            {surveyActive ? (
              <>
                <p className="survey-running-copy">
                  Hardware capture is {survey?.status === 'postprocessing' ? 'finished; generating AprilTag and MuJoCo artifacts' : 'running with adaptive camera centering and guarded telemetry'}.
                </p>
                {survey?.config && (
                  <div className="survey-summary">
                    <span>gaits <b>{survey.config.gaits.join(', ')}</b></span>
                    <span>speed <b>{survey.config.speed_mm_s} mm/s</b></span>
                    <span>pulse <b>{survey.config.direction_s} s</b></span>
                  </div>
                )}
                {!!survey?.log_tail.length && <pre className="survey-log">{survey.log_tail.slice(-5).join('\n')}</pre>}
                <button className="camera-power stop" disabled={busy} onClick={() => void stopSurvey()}>
                  {survey?.status === 'postprocessing' ? 'Stop post-processing' : 'Stop survey and limp'}
                </button>
              </>
            ) : showSurvey ? (
              <>
                <p>Walk selected scripted gaits forward and backward, recenter using AprilTags, and save hardware plus matched simulation data.</p>
                <button className="survey-collapse" onClick={() => setShowSurvey(false)}>Hide experiment controls</button>
                <div className="gait-picks">
                  {(survey?.gait_choices || []).map((gait) => (
                    <label key={gait.id} className={surveyGaits.includes(gait.id) ? 'selected' : ''}>
                      <input
                        type="checkbox"
                        checked={surveyGaits.includes(gait.id)}
                        onChange={() => toggleSurveyGait(gait.id)}
                      />
                      <b>{gait.id}</b>
                      <span>{gait.name}</span>
                    </label>
                  ))}
                </div>
                <div className="survey-fields">
                  <label>Speed mm/s<input type="number" min="5" max="40" value={surveySpeed} onChange={(event) => setSurveySpeed(event.target.value)} /></label>
                  <label>Each direction s<input type="number" min="1" max="20" value={surveyDuration} onChange={(event) => setSurveyDuration(event.target.value)} /></label>
                  <label>Recovery limit<input type="number" min="0" max="3" value={surveyRecoveries} onChange={(event) => setSurveyRecoveries(event.target.value)} /></label>
                </div>
                <div className="survey-policy">
                  <b>Recovery policy</b>
                  <span>Pre-trip warnings may pause → safe-zero → stand → retry. {survey?.hard_stop_policy}.</span>
                </div>
                <label className="motion-ack">
                  <input type="checkbox" checked={surveyAck} onChange={(event) => setSurveyAck(event.target.checked)} />
                  <span>I have cleared the area, can supervise the robot, and understand this starts physical motion.</span>
                </label>
                <button
                  className="survey-start full"
                  disabled={busy || collecting || !survey?.available || !state?.camera.enabled || safety?.verdict !== 'safe' || surveyGaits.length === 0 || !surveyAck}
                  onClick={() => void startSurvey()}
                >
                  {!survey?.available ? 'Connect robot in the Mac hub first' : !state?.camera.enabled ? 'Start camera for preflight' : safety?.verdict !== 'safe' ? 'Waiting for safe Vision/IMU preflight' : 'Start recorded gait survey'}
                </button>
              </>
            ) : (
              <div className="survey-collapsed">
                <p>Optional physical-motion experiment. Camera setup and visual calibration do not require this.</p>
                <button className="secondary full" onClick={() => setShowSurvey(true)}>Open gait survey</button>
              </div>
            )}

            {!surveyActive && survey?.status === 'complete' && (
              <div className="survey-result good">
                <b>Capture complete</b>
                <span>{Object.keys(survey.artifacts).length} artifacts saved</span>
                <code>{survey.run_dir}</code>
              </div>
            )}
            {!surveyActive && survey?.status === 'failed' && (
              <div className="survey-result bad">
                <b>Survey stopped safely</b>
                <span>{survey.error}</span>
                {survey.run_dir && <code>{survey.run_dir}</code>}
              </div>
            )}
          </section>

          <section className="panel camera-panel">
            <div className="panel-heading compact">
              <div>
                <div className="eyebrow">Input</div>
                <h2>Camera</h2>
              </div>
              <span className={`camera-state ${state?.camera.status}`}>{state?.camera.status || 'offline'}</span>
            </div>
            <div className="camera-picks">
              {cameraDevices.map((device) => (
                <button
                  key={device.index}
                  className={state?.camera.active_index === device.index || (!state?.camera.enabled && state?.camera.requested_index === device.index) ? 'selected' : ''}
                  disabled={busy || surveyActive || !device.available}
                  onClick={() => void switchCamera(device.index)}
                >
                  <span className="camera-name">{device.name}</span>
                  <span>{device.kind === 'continuity' ? 'iPhone Continuity Camera' : device.kind === 'built_in' ? 'Built-in camera' : device.kind === 'external' ? 'External camera' : `Camera ${device.index}`}</span>
                </button>
              ))}
            </div>
            <div className="camera-discovery">
              <span>
                {cameraDevices.length === 0
                  ? 'No camera is currently visible to macOS.'
                  : `${cameraDevices.length} camera${cameraDevices.length === 1 ? '' : 's'} detected by macOS.`}
              </span>
              <button className="secondary" disabled={busy || surveyActive} onClick={() => void rescanCameras()}>Rescan</button>
            </div>
            {cameraDevices.every((device) => device.kind !== 'continuity') && (
              <p className="camera-help">To use an iPhone, bring it near this Mac, lock it, and make sure Continuity Camera is enabled; then press Rescan.</p>
            )}
            {(state?.camera.error || state?.camera.scan_error) && (
              <div className="camera-error">{state.camera.error || state.camera.scan_error}</div>
            )}
            <div className={`capture-pipeline ${state?.camera.native_luma ? 'native' : ''}`}>
              <div>
                <span>Capture pipeline</span>
                <b>{state?.camera.native_luma ? 'Native luminance' : state?.camera.backend || 'Waiting'}</b>
              </div>
              <div>
                <span>Camera stream</span>
                <b>{state?.performance.capture_image_size_px?.join(' × ') || '—'} {state?.camera.pixel_format || ''}</b>
              </div>
              <small>
                {state?.camera.native_luma
                  ? 'AprilTags use the full-resolution Y plane; color conversion is limited to the preview and feet.'
                  : 'Auto mode uses native YUV when AVFoundation exposes it.'}
              </small>
            </div>
            <button
              className={state?.camera.enabled ? 'camera-power stop' : 'camera-power'}
              disabled={busy || surveyActive || (!state?.camera.enabled && cameraDevices.length === 0)}
              onClick={() => void setCameraEnabled(!state?.camera.enabled)}
            >
              {state?.camera.enabled ? 'Stop camera' : 'Start camera'}
            </button>
          </section>

          <section className="panel coverage-card">
            <div className="eyebrow">Direct visibility</div>
            <div className="coverage-row">
              <CoveragePill label="Robot tags" value={state?.coverage.robot_tags || 0} total={state?.coverage.robot_tags_required || 13} />
              <CoveragePill label="Floor tags" value={state?.coverage.floor_tags || 0} total={2} />
              <CoveragePill label="Feet" value={state?.coverage.feet || 0} total={6} />
            </div>
            <div className="tag-id-line">
              Seen IDs: {state?.coverage.robot_tag_ids.join(', ') || 'none'}
            </div>
          </section>

          <section className={`panel readiness-card ${readiness?.ready ? 'is-ready' : ''}`}>
            <div className="eyebrow">Calibration gate</div>
            <h2>{readiness?.headline || 'Waiting for vision'}</h2>
            <div className="progress-track">
              <div style={{width: `${(readiness?.progress || 0) * 100}%`}} />
            </div>
            <div className="progress-caption">
              <span>stable frames</span>
              <b>{readiness?.stable_frames || 0}/{readiness?.required_stable_frames || 12}</b>
            </div>
            {!!readiness?.blockers.length && (
              <ul className="issue-list blockers">
                {readiness.blockers.slice(0, 4).map((item) => <li key={item}>{item}</li>)}
              </ul>
            )}
            {!!readiness?.warnings.length && (
              <ul className="issue-list warnings">
                {readiness.warnings.map((item) => <li key={item}>{item}</li>)}
              </ul>
            )}
            <div className="scope-line">
              Capture scope <b>{readiness?.scope === 'lid_joints' ? '12 signed yaw/hip joints' : 'not established'}</b>
            </div>
          </section>

          <section className="panel capture-card">
            <div className="eyebrow">Stationary capture</div>
            <h2>{collecting ? 'Collecting calibration' : state?.calibration.status === 'complete' ? 'Capture complete' : 'Visual calibration'}</h2>
            {collecting ? (
              <>
                <div className="capture-count">
                  <b>{state?.calibration.accepted_frames}</b>
                  <span>of {state?.calibration.target_frames} accepted frames</span>
                </div>
                <div className="progress-track active">
                  <div style={{width: `${(state?.calibration.progress || 0) * 100}%`}} />
                </div>
                {state?.calibration.last_rejection && <p className="mini-warning">Paused: {state.calibration.last_rejection}</p>}
                <button className="secondary full" onClick={() => void cancelCalibration()}>Cancel capture</button>
              </>
            ) : (
              <>
                <p>Capture robust median vision-to-encoder offsets. Nothing is applied to the robot.</p>
                <button className="primary full" disabled={!readiness?.ready || busy || surveyActive} onClick={() => void startCalibration()}>
                  {readiness?.ready ? 'Start visual calibration' : 'Waiting for readiness'}
                </button>
              </>
            )}
          </section>
        </aside>
      </div>
    </main>
  )
}
