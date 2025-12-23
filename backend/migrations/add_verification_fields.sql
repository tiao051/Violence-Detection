-- Add Verification Fields to Detections Table
-- Migration Script for Camera Credibility System
-- Run this to add human verification tracking to existing detections table

-- Add verification columns
ALTER TABLE detections
ADD COLUMN verification_status VARCHAR(20) DEFAULT 'pending',
ADD COLUMN verified_by VARCHAR(100),
ADD COLUMN verified_at TIMESTAMP,
ADD COLUMN verification_notes TEXT;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_verification_status ON detections(verification_status);
CREATE INDEX IF NOT EXISTS idx_camera_verified ON detections(camera_id, verification_status);
CREATE INDEX IF NOT EXISTS idx_verified_at ON detections(verified_at);

-- Update existing rows to mark  as pending
UPDATE detections 
SET verification_status = 'pending'
WHERE verification_status IS NULL;

COMMENT ON COLUMN detections.verification_status IS 'Verification status: pending | true_positive | false_positive | uncertain';
COMMENT ON COLUMN detections.verified_by IS 'User ID who verified the alert';
COMMENT ON COLUMN detections.verified_at IS 'Timestamp when alert was verified';
COMMENT ON COLUMN detections.verification_notes IS 'Optional notes about verification decision';
