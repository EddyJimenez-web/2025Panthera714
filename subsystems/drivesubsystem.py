import math
import typing

import wpilib
import wpimath.units

from commands2 import Subsystem
from wpimath.filter import SlewRateLimiter
from wpimath.geometry import Pose2d, Rotation2d, Translation2d, Transform2d
from wpimath.kinematics import (
    ChassisSpeeds,
    SwerveModuleState,
    SwerveDrive4Kinematics,
    SwerveDrive4Odometry,
)
from wpilib import SmartDashboard, Field2d

from constants import DriveConstants, ModuleConstants
import swerveutils
from .maxswervemodule import MAXSwerveModule
from rev import SparkMax, SparkFlex
import navx


class DriveSubsystem(Subsystem):
    USE_IMU_POSE = True

    def __init__(self, maxSpeedScaleFactor=None) -> None:
        super().__init__()
        if maxSpeedScaleFactor is not None:
            assert callable(maxSpeedScaleFactor)

        self.maxSpeedScaleFactor = maxSpeedScaleFactor

        enabledChassisAngularOffset = 0 if DriveConstants.kAssumeZeroOffsets else 1

        # Create MAXSwerveModules
        self.frontLeft = MAXSwerveModule(
            DriveConstants.kFrontLeftDrivingCanId,
            DriveConstants.kFrontLeftTurningCanId,
            DriveConstants.kFrontLeftChassisAngularOffset * enabledChassisAngularOffset,
            turnMotorInverted=ModuleConstants.kTurningMotorInverted,
            motorControllerType=SparkFlex,
        )

        self.frontRight = MAXSwerveModule(
            DriveConstants.kFrontRightDrivingCanId,
            DriveConstants.kFrontRightTurningCanId,
            DriveConstants.kFrontRightChassisAngularOffset * enabledChassisAngularOffset,
            turnMotorInverted=ModuleConstants.kTurningMotorInverted,
            motorControllerType=SparkFlex,
        )

        self.rearLeft = MAXSwerveModule(
            DriveConstants.kRearLeftDrivingCanId,
            DriveConstants.kRearLeftTurningCanId,
            DriveConstants.kBackLeftChassisAngularOffset * enabledChassisAngularOffset,
            turnMotorInverted=ModuleConstants.kTurningMotorInverted,
            motorControllerType=SparkFlex,
        )

        self.rearRight = MAXSwerveModule(
            DriveConstants.kRearRightDrivingCanId,
            DriveConstants.kRearRightTurningCanId,
            DriveConstants.kBackRightChassisAngularOffset * enabledChassisAngularOffset,
            turnMotorInverted=ModuleConstants.kTurningMotorInverted,
            motorControllerType=SparkFlex,
        )

        # The gyro sensor
        self.gyro = navx.AHRS.create_spi()
        self._lastGyroPoseTime = 0
        self._lastGyroPose = (0.0, 0.0, 0.0)
        self._lastGyroState = "ok"

        # Slew rate filter variables for controlling lateral acceleration
        self.currentTranslationDir = 0.0
        self.currentTranslationMag = 0.0
        self.xSpeedDelivered = 0.0
        self.ySpeedDelivered = 0.0
        self.rotDelivered = 0.0

        self.magLimiter = SlewRateLimiter(DriveConstants.kMagnitudeSlewRate)
        self.rotLimiter = SlewRateLimiter(DriveConstants.kRotationalSlewRate)
        self.prevTime = wpilib.Timer.getFPGATimestamp()

        # Odometry class for tracking robot pose
        self.odometry = SwerveDrive4Odometry(
            DriveConstants.kDriveKinematics,
            Rotation2d(),
            (
                self.frontLeft.getPosition(),
                self.frontRight.getPosition(),
                self.rearLeft.getPosition(),
                self.rearRight.getPosition(),
            ),
        )
        self.odometryHeadingOffset = Rotation2d(0)
        self.imuToPose = Transform2d(0, 0, 0)

        self.resetOdometry(Pose2d(0, 0, 0))

        self.field = Field2d()
        SmartDashboard.putData("Field", self.field)

        self.simPhysics = None


    def periodic(self) -> None:
        if self.simPhysics is not None:
            self.simPhysics.periodic()

        # Update the odometry in the periodic block
        pose = self.odometry.update(
            self.getGyroHeading(),
            (
                self.frontLeft.getPosition(),
                self.frontRight.getPosition(),
                self.rearLeft.getPosition(),
                self.rearRight.getPosition(),
            ),
        )
        SmartDashboard.putNumber("x", pose.x)
        SmartDashboard.putNumber("y", pose.y)
        SmartDashboard.putNumber("heading", pose.rotation().degrees())
        self.field.setRobotPose(pose)


    def getHeading(self) -> Rotation2d:
        return self.getPose().rotation()


    def getPose(self) -> Pose2d:
        """Returns the currently-estimated pose of the robot.

        :returns: The pose.
        """
        if self.USE_IMU_POSE:
            return self.getImuPose().transformBy(self.imuToPose)
        else:
            return self.odometry.getPose()


    def resetOdometry(self, pose: Pose2d, resetGyro=True) -> None:
        """Resets the odometry to the specified pose.

        :param pose: The pose to which to set the odometry.
        :param resetGyro: Should the IMU (device) be reset too? (this will move the direction of swerve front)

        """
        if resetGyro:
            self.gyro.reset()
            self.gyro.resetDisplacement()
            self.gyro.setAngleAdjustment(0)
            self._lastGyroPose = (0.0, 0.0, 0.0)
            self._lastGyroPoseTime = 0

        # this line will have the effect we wanted *only* if gyro was connected at the time
        # (if it wasn't, probably need to move the deferred version of this logic to periodic())
        self.imuToPose = Transform2d(self.getImuPose(), pose)

        self.odometry.resetPosition(
            self.getGyroHeading(),
            (
                self.frontLeft.getPosition(),
                self.frontRight.getPosition(),
                self.rearLeft.getPosition(),
                self.rearRight.getPosition(),
            ),
            pose,
        )
        self.odometryHeadingOffset = self.odometry.getPose().rotation() - self.getGyroHeading()


    def adjustOdometry(self, dTrans: Translation2d, dRot: Rotation2d):
        # adjust the IMU position transform
        imuPose = self.getImuPose()
        oldImuPose = imuPose.transformBy(self.imuToPose)
        newImuPose = Pose2d(oldImuPose.translation() + dTrans, oldImuPose.rotation() + dRot)
        self.imuToPose = Transform2d(imuPose, newImuPose)

        # and adjust the odometry
        oldOdoPose = self.getPose()
        newOdoPose = Pose2d(oldOdoPose.translation() + dTrans, oldOdoPose.rotation() + dRot)
        self.odometry.resetPosition(
            oldOdoPose.rotation() - self.odometryHeadingOffset,
            (
                self.frontLeft.getPosition(),
                self.frontRight.getPosition(),
                self.rearLeft.getPosition(),
                self.rearRight.getPosition(),
            ),
            newOdoPose,
        )
        self.odometryHeadingOffset += dRot

    def stop(self):
        self.arcadeDrive(0, 0)

    def arcadeDrive(
        self,
        xSpeed: float,
        rot: float,
        assumeManualInput: bool = False,
    ) -> None:
        self.drive(xSpeed, 0, rot, False, False, square=assumeManualInput)

    def rotate(self, rotSpeed) -> None:
        """
        Rotate the robot in place, without moving laterally (for example, for aiming)
        :param rotSpeed: rotation speed
        """
        self.arcadeDrive(0, rotSpeed)

    def drive(
        self,
        xSpeed: float,
        ySpeed: float,
        rot: float,
        fieldRelative: bool,
        rateLimit: bool,
        square: bool = False
    ) -> None:
        """Method to drive the robot using joystick info.

        :param xSpeed:        Speed of the robot in the x direction (forward).
        :param ySpeed:        Speed of the robot in the y direction (sideways).
        :param rot:           Angular rate of the robot.
        :param fieldRelative: Whether the provided x and y speeds are relative to the
                              field.
        :param rateLimit:     Whether to enable rate limiting for smoother control.
        :param square:        Whether to square the inputs (useful for manual control)
        """

        if square:
            rot = rot * abs(rot)
            norm = math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed)
            xSpeed = xSpeed * norm
            ySpeed = ySpeed * norm

        if (xSpeed != 0 or ySpeed != 0) and self.maxSpeedScaleFactor is not None:
            norm = math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed)
            scale = abs(self.maxSpeedScaleFactor() / norm)
            if scale < 1:
                xSpeed = xSpeed * scale
                ySpeed = ySpeed * scale

        xSpeedCommanded = xSpeed
        ySpeedCommanded = ySpeed

        if rateLimit:
            # Convert XY to polar for rate limiting
            inputTranslationDir = math.atan2(ySpeed, xSpeed)
            inputTranslationMag = math.hypot(xSpeed, ySpeed)

            # Calculate the direction slew rate based on an estimate of the lateral acceleration
            if self.currentTranslationMag != 0.0:
                directionSlewRate = abs(
                    DriveConstants.kDirectionSlewRate / self.currentTranslationMag
                )
            else:
                directionSlewRate = 500.0
                # some high number that means the slew rate is effectively instantaneous

            currentTime = wpilib.Timer.getFPGATimestamp()
            elapsedTime = currentTime - self.prevTime
            angleDif = swerveutils.angleDifference(
                inputTranslationDir, self.currentTranslationDir
            )
            if angleDif < 0.45 * math.pi:
                self.currentTranslationDir = swerveutils.stepTowardsCircular(
                    self.currentTranslationDir,
                    inputTranslationDir,
                    directionSlewRate * elapsedTime,
                )
                self.currentTranslationMag = self.magLimiter.calculate(
                    inputTranslationMag
                )

            elif angleDif > 0.85 * math.pi:
                # some small number to avoid floating-point errors with equality checking
                # keep currentTranslationDir unchanged
                if self.currentTranslationMag > 1e-4:
                    self.currentTranslationMag = self.magLimiter.calculate(0.0)
                else:
                    self.currentTranslationDir = swerveutils.wrapAngle(
                        self.currentTranslationDir + math.pi
                    )
                    self.currentTranslationMag = self.magLimiter.calculate(
                        inputTranslationMag
                    )

            else:
                self.currentTranslationDir = swerveutils.stepTowardsCircular(
                    self.currentTranslationDir,
                    inputTranslationDir,
                    directionSlewRate * elapsedTime,
                )
                self.currentTranslationMag = self.magLimiter.calculate(0.0)

            self.prevTime = currentTime

            xSpeedCommanded = self.currentTranslationMag * math.cos(
                self.currentTranslationDir
            )
            ySpeedCommanded = self.currentTranslationMag * math.sin(
                self.currentTranslationDir
            )
            self.currentRotation = self.rotLimiter.calculate(rot)

        else:
            self.currentRotation = rot

        # Convert the commanded speeds into the correct units for the drivetrain
        self.xSpeedDelivered = xSpeedCommanded * DriveConstants.kMaxSpeedMetersPerSecond
        self.ySpeedDelivered = ySpeedCommanded * DriveConstants.kMaxSpeedMetersPerSecond
        self.rotDelivered = self.currentRotation * DriveConstants.kMaxAngularSpeed

        swerveModuleStates = DriveConstants.kDriveKinematics.toSwerveModuleStates(
            ChassisSpeeds.fromFieldRelativeSpeeds(
                self.xSpeedDelivered,
                self.ySpeedDelivered,
                self.rotDelivered,
                self.getGyroHeading(),
            )
            if fieldRelative
            else ChassisSpeeds(self.xSpeedDelivered, self.ySpeedDelivered, self.rotDelivered)
        )
        fl, fr, rl, rr = SwerveDrive4Kinematics.desaturateWheelSpeeds(
            swerveModuleStates, DriveConstants.kMaxSpeedMetersPerSecond
        )
        self.frontLeft.setDesiredState(fl)
        self.frontRight.setDesiredState(fr)
        self.rearLeft.setDesiredState(rl)
        self.rearRight.setDesiredState(rr)

    def setX(self) -> None:
        """Sets the wheels into an X formation to prevent movement."""
        self.frontLeft.setDesiredState(SwerveModuleState(0, Rotation2d.fromDegrees(45)))
        self.frontRight.setDesiredState(
            SwerveModuleState(0, Rotation2d.fromDegrees(-45))
        )
        self.rearLeft.setDesiredState(SwerveModuleState(0, Rotation2d.fromDegrees(-45)))
        self.rearRight.setDesiredState(SwerveModuleState(0, Rotation2d.fromDegrees(45)))

    def setModuleStates(
        self,
        desiredStates: typing.Tuple[
            SwerveModuleState, SwerveModuleState, SwerveModuleState, SwerveModuleState
        ],
    ) -> None:
        """Sets the swerve ModuleStates.

        :param desiredStates: The desired SwerveModule states.
        """
        fl, fr, rl, rr = SwerveDrive4Kinematics.desaturateWheelSpeeds(
            desiredStates, DriveConstants.kMaxSpeedMetersPerSecond
        )
        self.frontLeft.setDesiredState(fl)
        self.frontRight.setDesiredState(fr)
        self.rearLeft.setDesiredState(rl)
        self.rearRight.setDesiredState(rr)

    def resetEncoders(self) -> None:
        """Resets the drive encoders to currently read a position of 0."""
        self.frontLeft.resetEncoders()
        self.rearLeft.resetEncoders()
        self.frontRight.resetEncoders()
        self.rearRight.resetEncoders()


    def getGyroHeading(self) -> Rotation2d:
        """Returns the heading of the robot, tries to be smart when gyro is disconnected

        :returns: the robot's heading as Rotation2d
        """
        self.updateImuPose()
        return Rotation2d.fromDegrees(self._lastGyroPose[2] * DriveConstants.kGyroReversed)


    def getImuPose(self) -> Pose2d:
        """Returns the pose of the robot according to its gyro+accelerometer (IMU)

        :returns: the robot's pose as Pose2d
        """
        self.updateImuPose()
        p = self._lastGyroPose
        return Pose2d(p[0], p[1], wpimath.units.degreesToRadians(p[2]) * DriveConstants.kGyroReversed)


    def updateImuPose(self) -> None:
        now = wpilib.Timer.getFPGATimestamp()
        past = self._lastGyroPoseTime
        state = "ok"

        if not self.gyro.isConnected():
            state = "disconnected"
        else:
            if self.gyro.isCalibrating():
                state = "calibrating"
            self._lastGyroPose = (self.gyro.getDisplacementX(), self.gyro.getDisplacementY(), self.gyro.getAngle())
            self._lastGyroPoseTime = now

        if state != self._lastGyroState:
            SmartDashboard.putString("gyro", f"{state} after {int(now - past)}s")
            self._lastGyroState = state


    def getTurnRate(self) -> float:
        """Returns the turn rate of the robot (in degrees per second)

        :returns: The turn rate of the robot, in degrees per second
        """
        return self.gyro.getRate() * DriveConstants.kGyroReversed


    def getTurnRateDegreesPerSec(self) -> float:
        """Returns the turn rate of the robot (in degrees per second)

        :returns: The turn rate of the robot, in degrees per second
        """
        return self.getTurnRate() * 180 / math.pi


class BadSimPhysics(object):
    """
    this is the wrong way to do it, it does not scale!!!
    the right way is shown here: https://github.com/robotpy/examples/blob/main/Physics/src/physics.py
    and documented here: https://robotpy.readthedocs.io/projects/pyfrc/en/stable/physics.html
    (but for a swerve drive it will take some work to add correctly)
    """
    def __init__(self, drivetrain: DriveSubsystem, robot: wpilib.RobotBase):
        self.drivetrain = drivetrain
        self.robot = robot
        self.t = 0

    def periodic(self):
        past = self.t
        self.t = wpilib.Timer.getFPGATimestamp()
        if past == 0:
            return  # it was first time

        dt = self.t - past
        if self.robot.isEnabled():
            drivetrain = self.drivetrain

            states = (
                drivetrain.frontLeft.desiredState,
                drivetrain.frontRight.desiredState,
                drivetrain.rearLeft.desiredState,
                drivetrain.rearRight.desiredState,
            )
            speeds = DriveConstants.kDriveKinematics.toChassisSpeeds(states)

            dx = speeds.vx * dt
            dy = speeds.vy * dt

            heading = drivetrain.getHeading()
            trans = Translation2d(dx, dy).rotateBy(heading)
            rot = (speeds.omega * 180 / math.pi) * dt

            g = drivetrain.gyro
            g.setAngleAdjustment(g.getAngleAdjustment() + rot * DriveConstants.kGyroReversed)
            drivetrain.adjustOdometry(trans, Rotation2d())
