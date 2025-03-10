#
# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#

from __future__ import annotations

import math
import commands2

from commands.aimtodirection import AimToDirectionConstants
from commands.gotopoint import GoToPointConstants

from wpimath.geometry import Rotation2d, Translation2d
from wpilib import Timer, SmartDashboard

from constants import DriveConstants


class ApproachTag(commands2.Command):
    GLIDE_PATH_WIDTH_METERS = 0.025  # one inch tolerance for alignment
    OBJECT_SIZE_AT_FINAL_APPROACH_POINT = 9  # at the final approach point, object size on camera should be 9% of frame
    DESIRED_HEADING_RADIUS = 10  # if plus minus 10 degrees from desired heading, forward motion is allowed before final approach

    KPMULT_TRANSLATION = 0.5
    KPMULT_ROTATION = 0.3

    APPROACH_SHAPE_POWER = 2.0  # 2 = use parabola for before-final-approach trajectory shape, 3.0 = use cubic curve, etc.

    def __init__(
        self,
        camera,
        drivetrain,
        specificHeadingDegrees=None,
        speed=0.2,
        reverse=False,
        finalApproachSeconds=0.0,
        detectionTimeoutSeconds=2.0,
        cameraMinimumFps=4.0
    ):
        """
        Align the swerve robot to AprilTag precisely and then optionally slowly push it forward for a split second
        :param camera: camera to use, LimelightCamera or PhotonVisionCamera (from https://github.com/epanov1602/CommandRevSwerve/blob/main/docs/Adding_Camera.md)
        :param drivetrain: a drivetrain that implements swerve drive functionality (or mecanum/ball, must veer to sides)
        :param specificHeadingDegrees: do you want the robot to face in a very specific direction
        :param speed: if the camera is on the back of the robot, please use negative speed
        :param finalApproachSeconds: if you want the robot to push forward at the end of alignment
        :param reverse: set it =True if the camera is on the back of the robot (not front)
        :param detectionTimeoutSeconds: if no detection within this many seconds, assume the tag is lost
        :param cameraMinimumFps: what is the minimal number of **detected** frames per second expected from this camera
        """
        super().__init__()
        assert hasattr(camera, "getX"), "camera must have `getX()` to give us the object coordinate (in degrees)"
        assert hasattr(camera, "getA"), "camera must have `getA()` to give us object size (in % of screen)"
        assert hasattr(camera, "getSecondsSinceLastHeartbeat"), "camera must have a `getSecondsSinceLastHeartbeat()`"
        assert hasattr(drivetrain, "drive"), "drivetrain must have a `drive()` function, because we need a swerve drive"

        self.drivetrain = drivetrain
        self.camera = camera

        self.addRequirements(drivetrain)
        self.addRequirements(camera)

        self.reverse = reverse
        self.approachSpeed = min((1.0, abs(speed)))  # ensure that the speed is between 0.0 and 1.0
        self.finalApproachSeconds = finalApproachSeconds
        self.finalApproachSpeed = None
        self.tagToFinalApproachPt = None  # will be assigned in initialize()

        assert detectionTimeoutSeconds > 0, f"non-positive detectionTimeoutSeconds={detectionTimeoutSeconds}"
        self.detectionTimeoutSeconds = detectionTimeoutSeconds

        assert cameraMinimumFps > 0, f"non-positive cameraMinimumFps={cameraMinimumFps}"
        self.frameTimeoutSeconds = 1.0 / cameraMinimumFps

        # setting the target angle in a way that works for all cases
        self.targetDegrees = specificHeadingDegrees
        if specificHeadingDegrees is None:
            self.targetDegrees = lambda: self.drivetrain.getHeading().degrees()
        elif not callable(specificHeadingDegrees):
            self.targetDegrees = lambda: specificHeadingDegrees

        # state
        self.targetDirection = None
        self.lastSeenObjectTime = None
        self.lastSeenObjectX = 0.0
        self.lastSeenObjectSize = 0.0
        self.lastSeenDistanceToTag = None
        self.everSawObject = False
        self.tReachedGlidePath = 0.0  # time when aligned to the tag and desired direction for the first time
        self.tReachedFinalApproach = 0.0  # time when reached the final approach
        self.lostTag = ""
        self.finished = ""


    def initialize(self):
        self.targetDirection = Rotation2d.fromDegrees(self.targetDegrees())
        self.tReachedGlidePath = 0.0  # time when reached the glide path
        self.tReachedFinalApproach = 0.0  # time when reached the final approach
        self.lostTag = False
        self.lastSeenObjectX = 0.0
        self.lastSeenObjectSize = 0.0
        self.lastSeenDistanceToTag = 999
        self.lastSeenObjectTime = Timer.getFPGATimestamp()
        self.everSawObject = False
        self.finished = ""

        # final approach parameters
        self.tagToFinalApproachPt = self.computeTagDistanceFromTagSizeOnFrame(self.OBJECT_SIZE_AT_FINAL_APPROACH_POINT)
        self.finalApproachSpeed = self.computeProportionalSpeed(self.tagToFinalApproachPt) / self.APPROACH_SHAPE_POWER
        if self.finalApproachSpeed > self.approachSpeed:
            self.finalApproachSpeed = self.approachSpeed
        if self.finalApproachSpeed < GoToPointConstants.kMinTranslateSpeed:
            self.finalApproachSpeed = GoToPointConstants.kMinTranslateSpeed

        SmartDashboard.putString("command/c" + self.__class__.__name__, "running")


    def isFinished(self) -> bool:
        if self.finished:
            return True

        now = Timer.getFPGATimestamp()

        # bad ways to finish
        if self.lostTag:
            self.finished = self.lostTag
        elif now > self.lastSeenObjectTime + self.detectionTimeoutSeconds + self.finalApproachSeconds:
            delay = now - self.lastSeenObjectTime
            self.finished = f"not seen {int(1000 * delay)}ms"

        # good ways to finish
        elif self.tReachedFinalApproach != 0 and now > self.tReachedFinalApproach + self.finalApproachSeconds:
            self.finished = "approached" if self.finalApproachSeconds > 0 else "reached approach point"

        if not self.finished:
            return False

        SmartDashboard.putString("command/c" + self.__class__.__name__, self.finished)
        return True


    def end(self, interrupted: bool):
        self.drivetrain.arcadeDrive(0, 0)
        if interrupted:
            SmartDashboard.putString("command/c" + self.__class__.__name__, "interrupted")


    def execute(self):
        # 0. look at the camera
        now = Timer.getFPGATimestamp()
        self.updateVision(now)
        if self.lostTag:
            self.drivetrain.stop()
            return

        # 1. how many degrees are left to turn? (and recommended rotation speed)
        rotationSpeed, degreesLeftToRotate = self.getGyroBasedRotationSpeed()

        # 2. how far from the glide path? (and recommended translation speed)
        fwdSpeed, leftSpeed, distanceToGlidePath = self.getVisionBasedSwerveSpeed(now)

        # 3. have we reached the glide path?
        if self.hasReachedGlidePath(degreesLeftToRotate, distanceToGlidePath):
            if self.tReachedGlidePath == 0:
                self.tReachedGlidePath = now
                SmartDashboard.putString("command/c" + self.__class__.__name__, "reached glide path")

        # 4. be careful with forward speed
        if self.tReachedFinalApproach:
            # - if we are on final approach, completely ignore the fwdSpeed from the visual estimation
            completedPercentage = (now - self.tReachedFinalApproach) / self.finalApproachSeconds
            fwdSpeed = self.finalApproachSpeed * max((0.0, 1.0 - completedPercentage))
        elif now > self.lastSeenObjectTime + 0.5 * self.frameTimeoutSeconds:
            # - rare: if the object on the frame is not fresh, maybe it escaped to right or left, so move sideways only
            SmartDashboard.putString("command/c" + self.__class__.__name__, "tag not fresh, moving sideways")
            fwdSpeed = 0.0
        else:
            # - if we are very far from desired heading, scale that fwdSpeed down
            farFromDesiredHeading = abs(degreesLeftToRotate) / self.DESIRED_HEADING_RADIUS
            fwdSpeed *= max((0.0, 1.0 - farFromDesiredHeading))

        # 5. drive!
        if self.reverse:
            self.drivetrain.drive(-fwdSpeed, -leftSpeed, rotationSpeed, fieldRelative=False, rateLimit=False)
        else:
            self.drivetrain.drive(fwdSpeed, leftSpeed, rotationSpeed, fieldRelative=False, rateLimit=False)


    def getGyroBasedRotationSpeed(self):
        # 1. how many degrees are left to turn?
        currentDirection = self.drivetrain.getHeading()
        rotationRemaining = self.targetDirection - currentDirection
        degreesRemaining = rotationRemaining.degrees()
        # (optimize: do not turn left 350 degrees if you can just turn right -10 degrees, and vice versa)
        while degreesRemaining > 180:
            degreesRemaining -= 360
        while degreesRemaining < -180:
            degreesRemaining += 360

        # 2. proportional control: if we are almost finished turning, use slower turn speed (to avoid overshooting)
        proportionalSpeed = self.KPMULT_ROTATION * AimToDirectionConstants.kP * abs(degreesRemaining)
        if AimToDirectionConstants.kUseSqrtControl:
            proportionalSpeed = math.sqrt(0.5 * proportionalSpeed)  # will match the non-sqrt value when 50% max speed

        # 3. if target angle is on the right, we should really turn right (negative turn speed)
        turnSpeed = min([proportionalSpeed, 1.0])
        if degreesRemaining < 0:
            turnSpeed = -turnSpeed

        return turnSpeed, degreesRemaining


    def getVisionBasedSwerveSpeed(self, now):
        direction = self.getVisionBasedSwerveDirection(now)
        if direction is None:
            return 0.0, 0.0, None
        distance = direction.norm()

        # use proportional control to compute the velocity
        velocity = self.computeProportionalSpeed(distance)

        # distribute velocity between X and Y velocities in a way that gives us correct trajectory shape
        yVelocity = velocity * (direction.y / distance)
        xVelocity = velocity * (direction.x / distance) / self.APPROACH_SHAPE_POWER  # velocity must be smaller

        # do we need to rescale these velocities to meet constraints?
        norm = Translation2d(xVelocity, yVelocity).norm()
        if norm < GoToPointConstants.kMinTranslateSpeed:
            factor = GoToPointConstants.kMinTranslateSpeed / norm
            xVelocity *= factor
            yVelocity *= factor
        if norm > self.approachSpeed:
            factor = self.approachSpeed / norm
            xVelocity *= factor
            yVelocity *= factor

        # done
        return xVelocity, yVelocity, abs(direction.y)


    def getVisionBasedSwerveDirection(self, now):
        # can we trust the last seen object?
        if not (self.lastSeenObjectSize > 0):
            SmartDashboard.putString("command/c" + self.__class__.__name__, "never seen")
            return None  # the object is not yet there, hoping that this is temporary

        # where are we?
        robotX, robotY, tagX = self.localize()

        # have we reached the final approach point now? (must already be on glide path, otherwise it doesn't count)
        if self.tReachedGlidePath != 0 and self.tReachedFinalApproach == 0 and robotX > 0:
            SmartDashboard.putString("command/c" + self.__class__.__name__, "reached final approach")
            self.tReachedFinalApproach = now

        # if we already reached the glide path, go directly towards the tag (x, y = tagX, 0), otherwise towards 0,0
        if self.tReachedGlidePath != 0:
            direction = Translation2d(x=tagX - robotX, y=0.0 - robotY)
        else:
            direction = Translation2d(x=0.0 - robotX, y=0.0 - robotY)
        if not (direction.x != 0 or direction.y != 0):
            SmartDashboard.putString("command/c" + self.__class__.__name__, "distance not positive")
            return None

        return direction


    def localize(self):
        """
        localize the robot camera in the frame of the final approach point
        (i.e. final approach point is assumed to be at (x, y) = (0, 0), tag is assumed to be at (x, y) = (d, 0))
        :return: (x, y, d), where `x,y` are coordinates of the robot and `d` is distance between that point and tag
        """
        distanceToTag = self.lastSeenDistanceToTag

        # trigonometry: how many meters on the left is our tag? (if negative, then it's on the right)
        angle = Rotation2d.fromDegrees(self.lastSeenObjectX)
        y = -distanceToTag * angle.sin()

        distanceToFinalApproach = distanceToTag - self.tagToFinalApproachPt
        return -distanceToFinalApproach, -y, self.tagToFinalApproachPt


    def computeProportionalSpeed(self, distance) -> float:
        velocity = distance * GoToPointConstants.kPTranslate * self.KPMULT_TRANSLATION
        if GoToPointConstants.kUseSqrtControl:
            velocity = math.sqrt(0.5 * velocity * self.KPMULT_TRANSLATION)
        return velocity


    def computeTagDistanceFromTagSizeOnFrame(self, objectSizePercent):
        """
        # if a 0.2*0.2 meter AprilTag appears to take 1% of the screen on a 1.33-square-radian FOV camera...
        #   angular_area = area / distance^2
        #   4 * 0.01 = 0.2 * 0.2 / distance^2
        #   distance = sqrt(0.2 * 0.2 / (1.33 * 0.03)) = 1.0 meters
        # ... then it must be 1.0 meters away!
        #
        # in other words, we can use this approximate formula for distance (if we have 0.2 * 0.2 meter AprilTag)
        """
        return math.sqrt(0.2 * 0.2 / (1.33 * 0.01 * objectSizePercent))


    def hasReachedGlidePath(self, degreesLeftToRotate: float, distanceToGlidePath: float) -> bool:
        reachedNow = (
                abs(degreesLeftToRotate) < 2 * AimToDirectionConstants.kAngleToleranceDegrees and
                abs(distanceToGlidePath) < self.GLIDE_PATH_WIDTH_METERS
        )
        if self.tReachedGlidePath and not reachedNow:
            print(f"WARNING: not on glide path anymore (distance={distanceToGlidePath}, degrees={degreesLeftToRotate}")
        return reachedNow


    def updateVision(self, now):
        # non-sim logic:
        if self.camera.hasDetection():
            x = self.camera.getX()
            a = self.camera.getA()
            if x != 0 and a > 0:
                self.lastSeenDistanceToTag = self.computeTagDistanceFromTagSizeOnFrame(a)
                self.lastSeenObjectTime = now
                self.lastSeenObjectSize = a
                self.lastSeenObjectX = x
                self.everSawObject = True

        timeSinceLastHeartbeat = self.camera.getSecondsSinceLastHeartbeat()
        if timeSinceLastHeartbeat > self.frameTimeoutSeconds:
            self.lostTag = f"no camera heartbeat > {int(1000 * timeSinceLastHeartbeat)}ms"

        if self.lastSeenObjectTime != 0:
            timeSinceLastDetection = now - self.lastSeenObjectTime
            if timeSinceLastDetection > self.detectionTimeoutSeconds:
                if self.tReachedFinalApproach == 0:
                    self.lostTag = f"object lost for {int(1000 * timeSinceLastDetection)}ms before final approach"
                elif timeSinceLastDetection > self.detectionTimeoutSeconds + self.finalApproachSeconds:
                    self.lostTag = f"object lost for {int(1000 * timeSinceLastDetection)}ms on final approach"

