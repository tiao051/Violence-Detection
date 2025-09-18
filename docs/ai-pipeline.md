# AI Violence Detection Pipeline

A real-time AI pipeline for detecting violence in video streams with high accuracy and low latency.

## Pipeline Overview

The AI pipeline processes video input through three sequential phases to detect violence in real-time:

```
Video Input → Detection → Recognition → Classification → Violence Alert
```

**Performance Targets:**
- Response Time: <2 seconds
- Accuracy: >85% (Precision & Recall)
- Concurrent Processing: 10-20 video streams

## Phase 1: Detection

**Purpose**: Identify and locate persons and weapons in video frames

### What Detection Does:
- Extract bounding boxes for **persons** using YOLO/R-CNN
- Detect **weapons** (knives, guns, sticks) with type classification
- Filter out irrelevant objects (cars, furniture, etc.)
- Provide confidence scores for each detection

### Key Features:
- **Object Classification**: Determines if detected objects are human/weapon/other
- **Efficiency Focus**: Filters 80-90% of irrelevant regions early in pipeline
- **Multi-object Detection**: Handles multiple persons and weapons simultaneously

**Input**: Raw video frames
**Output**: Person bounding boxes + Weapon bounding boxes + Classifications + Confidence scores

---

## Phase 2: Recognition

**Purpose**: Track objects across frames and analyze relationships using multi-frame sequences

### Multi-frame Analysis Strategy:
- **Short sequences (4-6 frames)**: Immediate action detection
- **Medium sequences (8-16 frames)**: Complete action analysis
- **Long sequences (16-32 frames)**: Context and escalation patterns

### What Recognition Tracks:

#### Person-to-Person Relationships:
- Distance changes between people over time
- Approaching/retreating patterns
- Group formations and dispersals
- Interaction identification (who interacts with whom)

#### Movement Vector Analysis:
- **Speed changes**: Sudden acceleration (aggression indicators)
- **Direction changes**: Erratic vs. purposeful movements
- **Body posture evolution**: Standing → crouching → falling sequences
- **Limb movements**: Rapid arm/leg motions (potential strikes)

#### Weapon Relationships (when present):
- Person-weapon associations (who holds what)
- Weapon trajectory analysis (thrown, swung, pointed)
- Pickup/drop weapon events

### Key Violence Indicators Without Weapons:
1. **Rapid approach + contact**: Person A quickly moves toward Person B
2. **Impact patterns**: Sudden direction changes suggesting strikes/falls
3. **Defensive postures**: Arm positions indicating blocking/protection
4. **Group dynamics**: Multiple people converging on one person
5. **Pursuit patterns**: Chase behaviors

### Technologies Used:
- **Person Tracking**: DeepSORT or ByteTracker for consistent IDs
- **Pose Estimation**: MediaPipe Pose or AlphaPose
- **Motion Analysis**: Optical flow + LSTM networks

**Input**: Person/weapon detections from multiple sequential frames
**Output**: Person tracks + Movement patterns + Spatial relationships + Pose sequences

---

## Phase 3: Classification

**Purpose**: Make final violence/non-violence decision with confidence scoring

### Analysis Components:

#### 1. Action Pattern Classification
- **Individual actions**: Punch, kick, grab, push, fall, run, defensive posture
- **Interaction types**: Aggressive approach, mutual combat, chase, group attack
- **Action intensity**: Force and speed analysis
- **Action duration**: Quick strike vs. sustained struggle

#### 2. Temporal Context Analysis
- **Escalation patterns**: Does tension build over time?
- **De-escalation detection**: Are people separating/calming down?
- **Continuation prediction**: Will violence likely continue?
- **Sequence consistency**: Violence sustained across multiple frame windows

#### 3. Violence Severity Classification

**5-Level Classification System:**
- **Level 0 - Normal**: Regular daily activities
- **Level 1 - Suspicious**: Aggressive posturing, heated arguments, tension
- **Level 2 - Minor Violence**: Pushing, shoving, brief contact
- **Level 3 - Active Violence**: Punching, kicking, sustained fighting
- **Level 4 - Severe Violence**: Multiple attackers, weapons, serious injury risk

#### 4. Multi-Factor Decision Making

**Weighted Decision Factors:**
- Movement aggression (40%): Speed, force, targeting behavior
- Interaction patterns (30%): Approach, contact, retreat sequences
- Duration consistency (20%): Sustained vs. brief actions
- Context factors (10%): Environment, bystanders, weapons present

### Confidence Scoring:

**High Confidence Scenarios:**
- Clear strike patterns with visible impact
- Multiple violence indicators align consistently
- Sustained violent behavior across frame sequences

**Low Confidence Scenarios:**
- Single ambiguous action (could be sports/dance)
- Poor video quality or occlusion issues
- Edge cases (playful wrestling vs. real fighting)

### False Positive Reduction:

**Common False Positives Filtered:**
- Sports activities (boxing, martial arts training)
- Dancing or performance activities
- Playful interactions between people
- Accidental falls or collisions
- Medical emergencies (person collapsing)

**Filtering Methods:**
- Context learning from sports/dance training data
- Environment analysis (gym/stage detection)
- Crowd reaction analysis (cheering = likely sports)
- Equipment detection (boxing gloves, mats = training)

**Input**: Person tracks + Movement patterns + Spatial relationships
**Output**: Violence decision + Confidence score + Severity level + Scene description

---

## Final Output Structure

### Visual Output System

**Color-coded Bounding Box Display:**
- **🟢 Green (Safe)**: Normal behavior, no violence detected
- **🟡 Orange (Warning)**: Suspicious behavior, potential violence escalation  
- **🔴 Red (Violence)**: Confirmed violent behavior detected

**Important Note**: In rapidly escalating situations, the system may jump directly from Green → Red, bypassing the Warning stage.

### Output Formats by Input Type:

#### Real-time Sources (Phone camera/Webcam/IP Camera):
```
Live Video Stream + Real-time Bounding Box Overlay
- Persons: Colored boxes indicating current threat level
- Weapons: Additional weapon-type labels when detected
- Status indicator: Current system state (Safe/Warning/Violence)
- Timestamp: Real-time processing timestamp
```

#### Video File Processing:
```
Input Video → AI Processing → Output Video with Overlay
- Original video preserved with added bounding box visualization
- Color-coded boxes applied throughout entire video timeline
- Exportable processed video file (MP4/AVI format)
- Timeline markers showing violence detection events
```

#### IP Camera Playback Mode:
```
Historical Video Playback + Pre-processed Overlay
- Previously recorded footage with AI analysis overlay
- Scrubbing through timeline with violence markers
- Frame-by-frame analysis capability
- Export clips of specific incidents
```

### Technical Output Structure:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "frame_id": 1247,
  "detections": [
    {
      "person_id": "person_1",
      "bbox": [x1, y1, x2, y2],
      "status": "violence",  // safe, warning, violence
      "color_code": "#FF0000",  // red
      "confidence": 0.89,
      "actions": ["aggressive_approach", "strike_motion"]
    },
    {
      "person_id": "person_2", 
      "bbox": [x1, y1, x2, y2],
      "status": "warning",
      "color_code": "#FFA500",  // orange
      "confidence": 0.65,
      "actions": ["defensive_posture", "retreat"]
    }
  ],
  "weapons": [
    {
      "weapon_id": "weapon_1",
      "type": "knife",
      "bbox": [x1, y1, x2, y2],
      "associated_person": "person_1",
      "threat_level": "high"
    }
  ],
  "overall_scene_status": "violence",
  "recommended_action": "immediate_alert"
}
```

## Pipeline Flow Example

```
Frame Sequence: 0 → 5 → 10 → 15

Detection Phase:
Frame 0: 2 persons detected, no weapons
Frame 5: 2 persons detected, closer proximity  
Frame 10: 2 persons detected, very close contact
Frame 15: 2 persons detected, one person falling

Recognition Phase:
- Person A rapidly approached Person B (frames 0-5)
- Close contact with erratic movements (frames 5-10)
- Person B shows impact response/falling (frames 10-15)
- No weapons involved, pure person-to-person violence

Classification Phase:
- Action: Aggressive approach + impact + fall sequence
- Temporal: Escalation from normal → contact → violence
- Severity: Level 3 (Active Violence)
- Confidence: 0.87 (high confidence based on clear sequence)
- Decision: VIOLENCE DETECTED
```

This pipeline captures the temporal nature of violence through sequential frame analysis while maintaining real-time processing efficiency for immediate threat detection.