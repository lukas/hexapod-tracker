export type Vec2 = [number, number]

export interface Detection {
  tag_id: number
  label: string
  center_px: Vec2
  corners_px: Vec2[]
  source: 'detected' | 'optical_flow'
  confidence: number
  reprojection_rms_px: number
}

export interface FootTip {
  leg: number
  point_px: Vec2
  component_center_px: Vec2
  source: 'color' | 'optical_flow' | 'prediction'
  confidence: number
  occlusion_age_frames: number
}

export interface JointState {
  joint: string
  value_deg: number | null
  source: string
  confidence: number
  visual_deg: number | null
  visual_absolute_deg: number | null
  visual_source: string | null
  visual_confidence: number
  encoder_deg: number | null
  visual_minus_encoder_deg?: number
  visual_abs_minus_encoder_abs_deg?: number
}

export interface Readiness {
  ready: boolean
  status: string
  headline: string
  blockers: string[]
  warnings: string[]
  stable_frames: number
  required_stable_frames: number
  progress: number
  maximum_joint_motion_deg: number | null
  scope: 'none' | 'lid_joints' | 'full_zero_check'
  camera_calibration_provisional?: boolean
  coverage?: {
    robot_tags: number
    robot_tags_required: number
    floor_tags: number
    floor_tags_available: number
    feet: number
    signed_joints: number
  }
}

export interface CalibrationState {
  status: 'idle' | 'collecting' | 'complete' | 'cancelled'
  accepted_frames: number
  target_frames: number
  rejected_frames: number
  report_available: boolean
  progress?: number
  quality?: string
  report_path?: string
  last_rejection?: string
}

export interface GaitSurveyState {
  available: boolean
  active: boolean
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'postprocessing' | 'complete' | 'failed'
  run_dir: string | null
  error: string | null
  started_unix: number | null
  completed_unix: number | null
  config: {
    gaits: number[]
    speed_mm_s: number
    direction_s: number
    settle_s: number
    gait1_alpha: number
    adaptive_centering: boolean
    soft_recovery: boolean
    max_recoveries: number
  } | null
  artifacts: Record<string, string>
  gait_choices: Array<{id: number; name: string}>
  log_tail: string[]
  camera_note: string
  hard_stop_policy: string
}

export interface VisionState {
  ok: boolean
  service: string
  joint_frame?: 'robot_abs'
  joint_contract?: 'robot_abs_tibia_v2'
  camera: {
    enabled: boolean
    active_index: number | null
    requested_index: number
    indexes: number[]
    status: string
    error: string | null
    backend: string | null
    pixel_format: string | null
    native_luma: boolean
    capture_fps: number | null
    devices: Array<{
      index: number
      name: string
      kind: 'built_in' | 'continuity' | 'external' | 'configured' | 'camera'
      available: boolean
    }>
    scan_error: string | null
    scan_unix: number | null
    discovery_exact: boolean
  }
  performance: {
    fps: number
    frame_sequence: number
    frame_age_ms: number | null
    processing_width: number
    target_fps: number
    image_size_px: [number, number] | null
    capture_image_size_px: [number, number] | null
    detection_image_size_px: [number, number] | null
  }
  coverage: {
    robot_tags: number
    robot_tag_ids: number[]
    robot_tags_required: number
    floor_tags: number
    floor_tag_ids: number[]
    floor_tags_available: number
    feet: number
    foot_legs: number[]
  }
  readiness: Readiness
  calibration: CalibrationState
  survey: GaitSurveyState
  pose: {
    image_size_px: [number, number] | null
    tags: Detection[]
    feet: FootTip[]
    joints: JointState[]
    safety: {
      verdict: 'safe' | 'unsafe' | 'unverified'
      unsafe_reasons: string[]
      unknown_reasons: string[]
      warnings: string[]
      body_tilt_deg: number | null
      imu_tilt_deg: number | null
      motor_commands_sent: false
    } | null
    zero_check: {
      matches_zero: boolean
      out_of_tolerance: Array<{joint: string; error_deg: number}>
      issues: string[]
    } | null
    body_tilt_deg: number | null
    pose_reference: string | null
  }
  read_only: false
  motion_control_scope: 'acknowledged_guarded_gait_survey'
}

export interface CalibrationJoint {
  joint: string
  observable: boolean
  signed: boolean
  sample_count: number
  visual_minus_encoder_deg?: number
  visual_abs_minus_encoder_abs_deg?: number
  median_absolute_deviation_deg?: number
  median_confidence?: number
  quality: string
  interpretation: string
}

export interface CalibrationReport {
  created_unix: number
  sample_count: number
  quality: string
  camera_calibration_approximate: boolean
  signed_joint_count: number
  good_signed_joint_count: number
  joints: CalibrationJoint[]
  report_path: string
  advisory_only: true
  configuration_changed?: boolean
  applied_visual_bias_delta_deg?: Record<string, number>
  servo_zeros_changed: false
  motor_commands_sent: false
  next_action: string
}
