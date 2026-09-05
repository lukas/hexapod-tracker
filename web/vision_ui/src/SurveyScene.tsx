import {useMemo, useRef, useState} from 'react'
import type {PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent} from 'react'
import type {ZeroSurveyState, ZeroSurveyTag} from './types'

type Vec3 = [number, number, number]
type SceneScope = 'robot' | 'world'
type Projected = {x: number; y: number; depth: number}

const DEFAULT_YAW = -34
const DEFAULT_PITCH = 34

function mean(points: Vec3[]): Vec3 {
  if (!points.length) return [0, 0, 0]
  const total = points.reduce<Vec3>((sum, point) => [
    sum[0] + point[0], sum[1] + point[1], sum[2] + point[2],
  ], [0, 0, 0])
  return [total[0] / points.length, total[1] / points.length, total[2] / points.length]
}

function roleFor(tag: ZeroSurveyTag): 'robot' | 'floor' | 'extra' {
  if (tag.role === 'robot') return 'robot'
  if (tag.role === 'ground' || tag.role === 'calibration_anchor') return 'floor'
  return 'extra'
}

function addScaled(point: Vec3, axis: Vec3, distance: number): Vec3 {
  return [
    point[0] + axis[0] * distance,
    point[1] + axis[1] * distance,
    point[2] + axis[2] * distance,
  ]
}

function normalized(vector: Vec3): Vec3 {
  const length = Math.hypot(...vector)
  return length > 1e-8
    ? [vector[0] / length, vector[1] / length, vector[2] / length]
    : [1, 0, 0]
}

function tagCorners(tag: ZeroSurveyTag): Vec3[] {
  const center = tag.world_from_tag.translation_m
  const yAxis = normalized(tag.tag_y_world || [0, 1, 0])
  let xAxis = tag.tag_x_world
  if (!xAxis) {
    const normal = normalized(tag.tag_normal_world || [0, 0, 1])
    xAxis = normalized([
      yAxis[1] * normal[2] - yAxis[2] * normal[1],
      yAxis[2] * normal[0] - yAxis[0] * normal[2],
      yAxis[0] * normal[1] - yAxis[1] * normal[0],
    ])
  }
  const half = (tag.marker_size_m || 0.027) / 2
  return [
    addScaled(addScaled(center, xAxis, -half), yAxis, -half),
    addScaled(addScaled(center, xAxis, half), yAxis, -half),
    addScaled(addScaled(center, xAxis, half), yAxis, half),
    addScaled(addScaled(center, xAxis, -half), yAxis, half),
  ]
}

function median(values: number[]) {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

export default function SurveyScene({survey}: {survey: ZeroSurveyState}) {
  const [scope, setScope] = useState<SceneScope>('robot')
  const [yaw, setYaw] = useState(DEFAULT_YAW)
  const [pitch, setPitch] = useState(DEFAULT_PITCH)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState<[number, number]>([0, 0])
  const [selected, setSelected] = useState<number | null>(null)
  const drag = useRef<{x: number; y: number; yaw: number; pitch: number; pan: [number, number]; mode: 'orbit' | 'pan'; moved: boolean} | null>(null)
  const suppressedClick = useRef(false)

  const tags = survey.records.filter((tag) => tag.world_from_tag?.translation_m)
  const assignedRobotIds = new Set(
    survey.progress.robot_positions.flatMap((item) => (
      item.tag_id === null ? [] : [item.tag_id]
    )),
  )
  const displayRole = (tag: ZeroSurveyTag) => (
    assignedRobotIds.has(tag.tag_id) ? 'robot' as const : roleFor(tag)
  )
  const robotTags = tags.filter((tag) => displayRole(tag) === 'robot')
  const floorTags = tags.filter((tag) => displayRole(tag) === 'floor')
  // The metric floor anchors are part of the reconstruction, not optional
  // scene clutter. Only genuinely unassigned detections are hidden in the
  // focused view.
  const displayTags = scope === 'robot' ? [...robotTags, ...floorTags] : tags
  const selectedTag = tags.find((tag) => tag.tag_id === selected)
  const body = robotTags.find((tag) => tag.robot_frame === 'body')
  const hips = Array.from({length: 6}, (_, leg) => (
    robotTags.find((tag) => (
      tag.robot_frame === `L${leg}_coxa` && tag.kind === 'servo_lid'
    ))
  ))
  const knees = Array.from({length: 6}, (_, leg) => (
    robotTags.find((tag) => (
      tag.robot_frame === `L${leg}_femur`
      && tag.kind === 'servo_lid'
      && tag.joint === 'knee'
    ))
  ))

  const robotCenter = body?.world_from_tag.translation_m || mean(
    robotTags.map((tag) => tag.world_from_tag.translation_m),
  )
  const scenePoints = displayTags.map((tag) => tag.world_from_tag.translation_m)
  const worldCenter = mean(scenePoints)
  const center: Vec3 = scope === 'robot'
    ? [robotCenter[0], robotCenter[1], 0.085]
    : [worldCenter[0], worldCenter[1], Math.max(0.04, worldCenter[2] * 0.35)]

  const rotate = (point: Vec3): Projected => {
    const x = point[0] - center[0]
    const y = point[1] - center[1]
    const z = point[2] - center[2]
    const yawRad = yaw * Math.PI / 180
    const pitchRad = pitch * Math.PI / 180
    const x1 = Math.cos(yawRad) * x - Math.sin(yawRad) * y
    const y1 = Math.sin(yawRad) * x + Math.cos(yawRad) * y
    return {
      x: x1,
      y: -(Math.cos(pitchRad) * z - Math.sin(pitchRad) * y1),
      depth: Math.cos(pitchRad) * y1 + Math.sin(pitchRad) * z,
    }
  }

  const gridBounds = useMemo(() => {
    if (scope === 'robot') {
      const focusedPoints = [...robotTags, ...floorTags].map(
        (tag) => tag.world_from_tag.translation_m,
      )
      if (focusedPoints.length) {
        return {
          minX: Math.min(...focusedPoints.map((point) => point[0])) - 0.06,
          maxX: Math.max(...focusedPoints.map((point) => point[0])) + 0.06,
          minY: Math.min(...focusedPoints.map((point) => point[1])) - 0.06,
          maxY: Math.max(...focusedPoints.map((point) => point[1])) + 0.06,
        }
      }
      return {
        minX: robotCenter[0] - 0.24, maxX: robotCenter[0] + 0.24,
        minY: robotCenter[1] - 0.24, maxY: robotCenter[1] + 0.24,
      }
    }
    const ground = tags.filter((tag) => roleFor(tag) === 'floor')
    const points = ground.length ? ground.map((tag) => tag.world_from_tag.translation_m) : scenePoints
    return {
      minX: Math.min(...points.map((point) => point[0])) - 0.08,
      maxX: Math.max(...points.map((point) => point[0])) + 0.08,
      minY: Math.min(...points.map((point) => point[1])) - 0.08,
      maxY: Math.max(...points.map((point) => point[1])) + 0.08,
    }
  }, [scope, robotCenter[0], robotCenter[1], tags, scenePoints, robotTags, floorTags])

  const gridCorners: Vec3[] = [
    [gridBounds.minX, gridBounds.minY, 0],
    [gridBounds.maxX, gridBounds.minY, 0],
    [gridBounds.maxX, gridBounds.maxY, 0],
    [gridBounds.minX, gridBounds.maxY, 0],
  ]
  const cameraPath = scope === 'world' ? (survey.camera_path_m || []) : []
  const framePoints = [
    ...displayTags.flatMap(tagCorners),
    ...gridCorners,
    ...cameraPath.filter((_, index) => index % 4 === 0),
  ]
  const rotatedFrame = framePoints.map(rotate)
  const minX = Math.min(...rotatedFrame.map((point) => point.x))
  const maxX = Math.max(...rotatedFrame.map((point) => point.x))
  const minY = Math.min(...rotatedFrame.map((point) => point.y))
  const maxY = Math.max(...rotatedFrame.map((point) => point.y))
  const baseScale = Math.min(
    650 / Math.max(0.22, maxX - minX),
    340 / Math.max(0.16, maxY - minY),
  )
  const viewScale = baseScale * zoom
  const frameMidX = (minX + maxX) / 2
  const frameMidY = (minY + maxY) / 2
  const project = (point: Vec3) => {
    const rotated = rotate(point)
    return {
      x: 380 + pan[0] + (rotated.x - frameMidX) * viewScale,
      y: 226 + pan[1] + (rotated.y - frameMidY) * viewScale,
      depth: rotated.depth,
    }
  }

  const hipPoints = hips.flatMap((tag) => tag ? [tag.world_from_tag.translation_m] : [])
  const chassisZ = Math.max(0.025, median(hipPoints.map((point) => point[2])) - 0.022)
  const chassis = body && hipPoints.length >= 3
    ? [...hipPoints]
      .sort((a, b) => (
        Math.atan2(a[1] - robotCenter[1], a[0] - robotCenter[0])
        - Math.atan2(b[1] - robotCenter[1], b[0] - robotCenter[0])
      ))
      .map<Vec3>((point) => [
        robotCenter[0] + (point[0] - robotCenter[0]) * 0.72,
        robotCenter[1] + (point[1] - robotCenter[1]) * 0.72,
        chassisZ,
      ])
    : []

  const gridLines: Array<[Vec3, Vec3]> = []
  const step = scope === 'robot' ? 0.05 : 0.1
  for (let x = Math.ceil(gridBounds.minX / step) * step; x <= gridBounds.maxX; x += step) {
    gridLines.push([[x, gridBounds.minY, 0], [x, gridBounds.maxY, 0]])
  }
  for (let y = Math.ceil(gridBounds.minY / step) * step; y <= gridBounds.maxY; y += step) {
    gridLines.push([[gridBounds.minX, y, 0], [gridBounds.maxX, y, 0]])
  }

  const targetPosition = survey.progress.robot_positions.find(
    (item) => item.position === survey.guidance?.target_position,
  )
  const targetPoint = targetPosition?.expected_world_position_m || null
  const resetView = () => {
    setYaw(DEFAULT_YAW)
    setPitch(DEFAULT_PITCH)
    setZoom(1)
    setPan([0, 0])
  }
  const onPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const panMode = event.shiftKey || event.button === 1 || event.button === 2
    drag.current = {x: event.clientX, y: event.clientY, yaw, pitch, pan, mode: panMode ? 'pan' : 'orbit', moved: false}
    suppressedClick.current = false
  }
  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!drag.current) return
    const dx = event.clientX - drag.current.x
    const dy = event.clientY - drag.current.y
    drag.current.moved ||= Math.hypot(dx, dy) > 3
    suppressedClick.current = drag.current.moved
    if (drag.current.mode === 'pan') {
      setPan([drag.current.pan[0] + dx, drag.current.pan[1] + dy])
    } else {
      setYaw(drag.current.yaw - dx * 0.36)
      setPitch(Math.max(4, Math.min(86, drag.current.pitch - dy * 0.32)))
    }
  }
  const onPointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    drag.current = null
  }
  const onWheel = (event: ReactWheelEvent<SVGSVGElement>) => {
    event.preventDefault()
    setZoom((current) => Math.max(0.45, Math.min(4.0, current * Math.exp(-event.deltaY * 0.0015))))
  }

  if (!tags.length) {
    return (
      <div className="survey-map empty-map">
        <div className="map-heading"><div><span>Live reconstruction</span><b>Interactive 3D robot</b></div><em>Awaiting origin lock</em></div>
        <svg viewBox="0 0 760 440" role="img" aria-label="3D robot survey awaiting tag measurements">
          <path className="map-floor" d="M90 328 L380 170 L680 330 L390 426 Z" />
          <g className="ghost-robot"><path d="M315 248 L352 219 L416 221 L452 250 L416 282 L351 280 Z" /></g>
          <text x="380" y="95" textAnchor="middle">Measured robot geometry will appear here</text>
        </svg>
      </div>
    )
  }

  const floorPolygon = gridCorners.map((point) => {
    const projected = project(point)
    return `${projected.x},${projected.y}`
  }).join(' ')
  const chassisPolygon = chassis.map((point) => {
    const projected = project(point)
    return `${projected.x},${projected.y}`
  }).join(' ')
  const pathPoints = cameraPath.map((point) => {
    const projected = project(point)
    return `${projected.x},${projected.y}`
  }).join(' ')
  const axisOrigin: Vec3 = [gridBounds.minX + 0.035, gridBounds.minY + 0.035, 0.006]
  const axisLength = scope === 'robot' ? 0.065 : 0.1
  const axes: Array<{name: string; end: Vec3}> = [
    {name: 'x', end: addScaled(axisOrigin, [1, 0, 0], axisLength)},
    {name: 'y', end: addScaled(axisOrigin, [0, 1, 0], axisLength)},
    {name: 'z', end: addScaled(axisOrigin, [0, 0, 1], axisLength)},
  ]
  const projectedOrigin = project(axisOrigin)

  return (
    <div className="survey-map survey-scene">
      <div className="map-heading">
        <div><span>Measured reconstruction</span><b>{scope === 'robot' ? 'Robot + metric floor tags' : 'Robot, floor, extras + walk path'}</b></div>
        <div className="scene-controls">
          <div className="map-view-toggle">
            <button className={scope === 'robot' ? 'active' : ''} onClick={() => { setScope('robot'); setZoom(1); setPan([0, 0]) }}>Robot + floor</button>
            <button className={scope === 'world' ? 'active' : ''} onClick={() => { setScope('world'); setZoom(1) }}>Full scan</button>
          </div>
          <button className="scene-reset zoom-button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.45, value / 1.2))}>−</button>
          <button className="scene-reset zoom-button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(4, value * 1.2))}>+</button>
          <button className="scene-reset" onClick={resetView}>Reset view</button>
        </div>
      </div>
      <svg
        viewBox="0 0 760 440"
        role="img"
        aria-label="Interactive three dimensional reconstruction of the measured hexapod and AprilTags"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onDoubleClick={resetView}
        onContextMenu={(event) => event.preventDefault()}
      >
        <defs>
          <linearGradient id="sceneFloor" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#f7fbff"/><stop offset="1" stopColor="#eaf3fb"/></linearGradient>
          <linearGradient id="chassisTop" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#304a60"/><stop offset="1" stopColor="#1d3346"/></linearGradient>
          <filter id="sceneShadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#17324d" floodOpacity=".20"/></filter>
        </defs>
        <rect x="1" y="1" width="758" height="438" rx="18" className="scene-sky" />
        <polygon className="scene-floor-plane" points={floorPolygon} />
        {gridLines.map(([start, end], index) => {
          const a = project(start); const b = project(end)
          return <line key={index} className="scene-grid-line" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
        })}
        {pathPoints && <polyline className="phone-path" points={pathPoints} />}
        {chassisPolygon && <polygon className="robot-chassis" points={chassisPolygon} filter="url(#sceneShadow)" />}
        {hips.map((hip, leg) => {
          const knee = knees[leg]
          if (!hip) return null
          const hipPoint = project(hip.world_from_tag.translation_m)
          const bodyPoint = body ? project(body.world_from_tag.translation_m) : null
          const kneePoint = knee ? project(knee.world_from_tag.translation_m) : null
          return (
            <g key={leg} className="robot-leg">
              {bodyPoint && <line className="coxa-link" x1={bodyPoint.x} y1={bodyPoint.y} x2={hipPoint.x} y2={hipPoint.y} />}
              {kneePoint && <line className="femur-link" x1={hipPoint.x} y1={hipPoint.y} x2={kneePoint.x} y2={kneePoint.y} />}
              <circle className="joint-hub" cx={hipPoint.x} cy={hipPoint.y} r="7" />
              {kneePoint && <circle className="joint-hub knee" cx={kneePoint.x} cy={kneePoint.y} r="6" />}
              <text className="leg-label" x={hipPoint.x + 9} y={hipPoint.y + 17}>L{leg}</text>
            </g>
          )
        })}
        {targetPoint && (() => {
          const target = project(targetPoint)
          return <g className="target-beacon" transform={`translate(${target.x} ${target.y})`}><circle r="19"/><circle r="7"/><text x="25" y="5">Next: {targetPosition?.position}</text></g>
        })()}
        {[...displayTags].sort((a, b) => rotate(a.world_from_tag.translation_m).depth - rotate(b.world_from_tag.translation_m).depth).map((tag) => {
          const point = tag.world_from_tag.translation_m
          const centerPoint = project(point)
          const role = displayRole(tag)
          const corners = tagCorners(tag).map(project)
          const cornerPoints = corners.map((corner) => `${corner.x},${corner.y}`).join(' ')
          const orientation = tag.tag_y_world
          const arrow = orientation ? project(addScaled(point, normalized(orientation), 0.045)) : null
          const showLabel = scope === 'robot' || role !== 'extra' || selected === tag.tag_id
          return (
            <g
              key={tag.tag_id}
              className={`map-tag ${role} ${tag.stable ? 'stable' : 'warming'} ${selected === tag.tag_id ? 'selected' : ''} ${survey.guidance?.target_tag_id === tag.tag_id ? 'targeted' : ''}`}
              onClick={() => {
                if (!suppressedClick.current) setSelected(tag.tag_id)
                suppressedClick.current = false
              }}
            >
              {arrow && role !== 'extra' && <line className="orientation-arrow" x1={centerPoint.x} y1={centerPoint.y} x2={arrow.x} y2={arrow.y} />}
              <polygon points={cornerPoints} filter="url(#sceneShadow)" />
              <circle cx={centerPoint.x} cy={centerPoint.y} r="2.8" />
              {showLabel && <text x={centerPoint.x + 12} y={centerPoint.y - 10}>#{tag.tag_id}{scope === 'robot' && tag.robot_frame ? ` · ${tag.robot_frame}${tag.mount_side ? ` ${tag.mount_side}` : ''}` : ''}</text>}
              <title>{`${tag.label || role} · ${tag.observations} observations`}</title>
            </g>
          )
        })}
        {survey.camera_position_m && scope === 'world' && (() => {
          const camera = project(survey.camera_position_m)
          return <g className="phone-marker" transform={`translate(${camera.x} ${camera.y})`}><path d="M0 -10 L8 9 L0 6 L-8 9 Z"/><text x="13" y="5">iPhone</text></g>
        })()}
        <g className="map-axis">
          {axes.map((axis) => {
            const end = project(axis.end)
            return <g key={axis.name}><line x1={projectedOrigin.x} y1={projectedOrigin.y} x2={end.x} y2={end.y}/><text x={end.x + 5} y={end.y + 4}>{axis.name}</text></g>
          })}
        </g>
        <g className="orbit-hint"><path d="M24 30 C38 16 62 16 76 30"/><path d="M72 22 L77 30 L68 31"/><text x="24" y="49">drag: orbit · shift-drag: pan · wheel: zoom · double-click: reset</text></g>
      </svg>
      <div className="map-legend"><span><i className="robot" />Robot mount</span><span><i className="floor" />Floor tag</span><span><i className="extra" />Unassigned detection</span><span className="legend-arrow">↗ measured tag +Y</span></div>
      <div className="scene-note">Blue markers are measured metric floor references; green markers are robot mounts. Geometry is drawn only between measured references.</div>
      {selectedTag && <div className="selected-tag"><b>Tag #{selectedTag.tag_id}</b><span>{selectedTag.robot_frame || selectedTag.label || roleFor(selectedTag)}</span><span>{selectedTag.observations} views · {selectedTag.translation_spread_mm?.toFixed(1) || '—'} mm spread · {selectedTag.rotation_spread_deg?.toFixed(1) || '—'}°</span></div>}
    </div>
  )
}
