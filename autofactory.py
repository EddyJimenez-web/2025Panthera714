#
# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#

from __future__ import annotations

import math

from wpilib import SendableChooser, SmartDashboard
from wpimath.geometry import Translation2d, Rotation2d, Pose2d
from commands2 import TimedCommandRobot, WaitCommand, InstantCommand, Command

import constants
import autowaypoints
from commands.aimtodirection import AimToDirection
from commands.jerky_trajectory import JerkyTrajectory, SwerveTrajectory, mirror
from commands.intakecommands import IntakeGamepiece, AssumeIntakeLoaded
from commands.setcamerapipeline import SetCameraPipeline
from commands.swervetopoint import SwerveMove
from commands.reset_xy import ResetXY

# which trajectory to use
BACKUP_METERS = 0.2

class AutoFactory(object):

    @staticmethod
    def makeAutoCommand(self):
        startPos = self.startPos.getSelected()
        startX, startY, startHeading = startPos
        startPosCmd = ResetXY(startX, startY, startHeading, drivetrain=self.robotDrive)

        goal1traj = self.goal1traj.getSelected()
        goal1branch = self.goal1branch.getSelected()
        goal1height = self.goal1height.getSelected()

        goal2height = self.goal2height.getSelected()
        if goal2height == "same": goal2height = goal1height

        trajectoryClass, swerve = self.autoTrajStyle.getSelected()

        drivingSpeed = self.autoDrvSpeed.getSelected()  # 0.2 default

        # commands for approaching and retreating from goal 1 scoring location
        headingTags1, approachCmd, retreatCmd, take2Cmd, headingTags2, feeder = goal1traj(
            self, startPos, speed=drivingSpeed, branch=goal1branch, swerve=swerve, TrajectoryCommand=trajectoryClass
        )

        # if we are allowed to use rearview camera, can the `retreatCmd` be smarter?
        retreatCmd = AutoFactory.approachFeeder(
            self, self.rearCamera, feeder.location[2], traj=retreatCmd, tags=feeder.tags, speed=drivingSpeed
        )

        # command do we use for aligning the robot to AprilTag after approaching goal 1
        approachCmd = AutoFactory.approachReef(
            self, traj=approachCmd, headingTags=headingTags1, branch=goal1branch, height=goal1height
        )

        shootCmd = AutoFactory.moveArm(self, height=goal1height, final=True).andThen(
            AutoFactory.ejectGamepiece(self, calmdownSecondsBeforeFiring=0.0)
        ).andThen(
            AutoFactory.moveArm(self, height=goal1height, final=False)
        )
        backupCmd = SwerveMove(metersToTheLeft=0, metersBackwards=BACKUP_METERS, drivetrain=self.robotDrive, slowDownAtFinish=False)
        dropArmCmd = AutoFactory.moveArm(self, height="intake").andThen(
            AutoFactory.moveArm(self, height="intake")  # belt-and-suspenders hack
        )

        # commands for reloading a new gamepiece from the feeding station
        armToIntakePositionCmd = AutoFactory.moveArm(self, height="intake")
        intakeCmd = AutoFactory.intakeGamepiece(self, speed=0.115)  # .onlyIf(armToIntakePositionCmd.succeeded)
        reloadCmd = armToIntakePositionCmd.andThen(intakeCmd)

        # commands for aligning with the second tag
        take2Cmd = AutoFactory.approachReef(
            self, traj=take2Cmd, headingTags=headingTags2, branch=goal1branch, height=goal2height, speed=1.0
        )

        # commands for scoring that second gamepiece
        shoot2Cmd = AutoFactory.moveArm(self, height=goal2height, final=True).andThen(
            AutoFactory.ejectGamepiece(self, calmdownSecondsBeforeFiring=0)
        )
        backup2Cmd = SwerveMove(metersToTheLeft=0, metersBackwards=BACKUP_METERS, drivetrain=self.robotDrive)
        dropArm2Cmd = AutoFactory.moveArm(self, height="intake")

        # connect them all (and report status in "autoStatus" widget at dashboard)
        result = startPosCmd.andThen(
            runCmd("intake loaded...", AssumeIntakeLoaded(self.intake))  # tell the robot to assume intake loaded
        ).andThen(
            runCmd("approach...", approachCmd)
        ).andThen(
            runCmd("shoot...", shootCmd)
        ).andThen(
            runCmd("backup...", backupCmd)
        ).andThen(
            runCmd("retreat...", retreatCmd.alongWith(dropArmCmd))
        ).andThen(
            runCmd("reload...", reloadCmd)
        ).andThen(
            runCmd("take2...", take2Cmd)
        ).andThen(
            runCmd("shoot2...", shoot2Cmd)
        ).andThen(
            runCmd("backup2...", backup2Cmd.andThen(dropArm2Cmd))
        ).andThen(
            autoStatus("done")
        )

        return result

    @staticmethod
    def init(self):
        SmartDashboard.putString("autoStatus", "initialized")

        # 0. starting position for all autos
        self.startPos = SendableChooser()
        self.startPos.addOption("1: L+", (7.189, 7.75, -180))  # (x, y, headingDegrees)
        self.startPos.addOption("2: L", (6.98, 6.177, -142))  # (x, y, headingDegrees)
        self.startPos.setDefaultOption("3: ML", (7.189, 4.40, 180))  # (x, y, headingDegrees)
        self.startPos.addOption("4: MID", (7.189, 4.025, 180))  # (x, y, headingDegrees)
        self.startPos.addOption("5: MR", (7.189, 3.65, 180))  # (x, y, headingDegrees)
        self.startPos.addOption("6: R", (6.98, 1.897, 142))  # (x, y, headingDegrees)
        self.startPos.addOption("7: R+", (7.189, 0.4, 180))  # (x, y, headingDegrees)

        # goal 1
        #  - which reef to choose for goal 1
        self.goal1traj = SendableChooser()
        self.goal1traj.addOption("C", AutoFactory.trajectoriesToSideC)
        self.goal1traj.setDefaultOption("D<", AutoFactory.trajectoriesToSideDLeft)
        self.goal1traj.addOption("D>", AutoFactory.trajectoriesToSideDRight)
        self.goal1traj.addOption("E", AutoFactory.trajectoriesToSideE)
        self.goal1traj.addOption("F", AutoFactory.trajectoriesToSideF)

        # - which branch to choose for goal 1
        self.goal1branch = SendableChooser()
        self.goal1branch.setDefaultOption("left", "left")
        self.goal1branch.addOption("right", "right")

        # - which scoring level to choose for goal 1
        self.goal1height = SendableChooser()
        self.goal1height.addOption("base", "base")
        self.goal1height.addOption("level 2", "level 2")
        self.goal1height.addOption("level 3", "level 3")
        self.goal1height.setDefaultOption("level 4", "level 4")

        # goal 2
        # - which scoring level to choose for goal 2
        self.goal2height = SendableChooser()
        self.goal2height.setDefaultOption("same", "same")
        self.goal2height.addOption("base", "base")
        self.goal2height.addOption("level 2", "level 2")
        self.goal2height.addOption("level 3", "level 3")
        self.goal2height.addOption("level 4", "level 4")

        # how to drive between waypoints? (like a tank, like a frog, or what)
        self.autoTrajStyle = SendableChooser()
        self.autoTrajStyle.addOption("tank", (JerkyTrajectory, "last-point"))
        self.autoTrajStyle.setDefaultOption("rabbit", (JerkyTrajectory, True))
        self.autoTrajStyle.addOption("snowboard", (SwerveTrajectory, True))

        # driving speed
        self.autoDrvSpeed = SendableChooser()
        self.autoDrvSpeed.addOption("0.2", 0.2)
        self.autoDrvSpeed.addOption("0.3", 0.3)
        self.autoDrvSpeed.addOption("0.45", 0.45)
        self.autoDrvSpeed.addOption("0.7", 0.7)
        self.autoDrvSpeed.setDefaultOption("1.0", 1.0)

        SmartDashboard.putData("auto1StartPos", self.startPos)
        SmartDashboard.putData("auto2Paths", self.goal1traj)
        SmartDashboard.putData("auto3Branch", self.goal1branch)
        SmartDashboard.putData("auto4Scoring1", self.goal1height)
        SmartDashboard.putData("auto5Scoring2", self.goal2height)
        SmartDashboard.putData("auto6TrjStyle", self.autoTrajStyle)
        SmartDashboard.putData("auto9DriveSpd", self.autoDrvSpeed)

        self.startPos.onChange(lambda _: AutoFactory.updateDashboard(self))
        self.goal1traj.onChange(lambda _: AutoFactory.updateDashboard(self))
        self.goal1branch.onChange(lambda _: AutoFactory.updateDashboard(self))


    @staticmethod
    def trajectoriesToSideDLeft(self, start, branch="right", speed=0.2, swerve="last-point", TrajectoryCommand=JerkyTrajectory):
        assert branch in ("right", "left")

        endpoint = autowaypoints.SideDLeft.kEndpoint[branch]
        headingTags = endpoint[2], autowaypoints.SideDLeft.tags
        feeder = constants.LeftFeeder

        approach = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=speed,
            waypoints=[start] + autowaypoints.SideDLeft.kApproach,
            endpoint=endpoint,
            stopAtEnd=False,
        )

        retreat = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=-speed,
            waypoints=[endpoint] + autowaypoints.SideDLeft.kReload,
            endpoint=None, #feeder.location,
        )

        take2, headingTags2 = AutoFactory.goToSideF(self, branch, speed, swerve, TrajectoryCommand)

        return headingTags, approach, retreat, take2, headingTags2, feeder



    @staticmethod
    def trajectoriesToSideDRight(self, start, branch="right", speed=0.2, swerve="last-point", TrajectoryCommand=JerkyTrajectory):
        assert branch in ("right", "left")

        endpoint = autowaypoints.SideDRight.kEndpoint[branch]
        feeder = constants.RightFeeder
        headingTags = endpoint[2], autowaypoints.SideDRight.tags

        approach = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=speed,
            waypoints=[start] + autowaypoints.SideDRight.kApproach,
            endpoint=endpoint,
            stopAtEnd=False,
        )

        retreat = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=-speed,
            waypoints=[endpoint] + autowaypoints.SideDRight.kReload,
            endpoint=None, #feeder.location
        )

        take2, headingTags2 = AutoFactory.goToSideB(self, branch, speed, swerve, TrajectoryCommand)

        return headingTags, approach, retreat, take2, headingTags2, feeder


    @staticmethod
    def trajectoriesToSideC(self, start, branch="right", speed=0.2, swerve="last-point", TrajectoryCommand=JerkyTrajectory):
        assert branch in ("right", "left")

        # make the mirror image of side E

        endpoint = autowaypoints.SideC.kEndpoint[branch]
        feeder = constants.RightFeeder
        headingTags = endpoint[2], autowaypoints.SideC.tags

        approach = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=speed,
            waypoints=[start] + autowaypoints.SideC.kApproach,
            endpoint=endpoint,
            stopAtEnd=False,
        )

        retreat = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=-speed,
            waypoints=[endpoint] + autowaypoints.SideC.kReload,
            endpoint=None, #feeder.location
        )

        take2, headingTags2 = AutoFactory.goToSideB(self, branch, speed, swerve, TrajectoryCommand)

        return headingTags, approach, retreat, take2, headingTags2, feeder


    @staticmethod
    def trajectoriesToSideE(self, start, branch="right", speed=0.2, swerve="last-point", TrajectoryCommand=JerkyTrajectory):
        assert branch in ("right", "left")

        endpoint = autowaypoints.SideE.kEndpoint[branch]
        feeder = constants.LeftFeeder
        headingTags = endpoint[2], autowaypoints.SideE.tags

        approach = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=speed,
            waypoints=[start] + autowaypoints.SideE.kApproach,
            endpoint=endpoint,
            stopAtEnd=False,
        )

        retreat = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=-speed,
            waypoints=[endpoint] + autowaypoints.SideE.kReload,
            endpoint=None, #feeder.location
        )

        take2, headingTags2 = AutoFactory.goToSideF(self, branch, speed, swerve, TrajectoryCommand)

        return headingTags, approach, retreat, take2, headingTags2, feeder



    @staticmethod
    def trajectoriesToSideF(self, start, branch="right", speed=0.2, swerve="last-point", TrajectoryCommand=JerkyTrajectory):
        assert branch in ("right", "left")

        headingTags = -60, (6, 19)
        endpoint = (3.070, 6.146, -60.0) if branch == "right" else (3.050, 6.306, -60.0)
        feeder = constants.LeftFeeder

        approach = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=speed,
            waypoints=[
                start,
                (5.991, 6.146, -180.0),
                (4.991, 6.346, -60.0),
            ],
            endpoint=endpoint,
        )

        retreat = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            speed=-speed,
            waypoints=[
                endpoint,
                (2.085, 6.215, -54.0),
            ],
            endpoint=None, #feeder.location
        )

        take2, headingTags2 = AutoFactory.goToSideF(self, branch, speed, swerve, TrajectoryCommand)

        return headingTags, approach, retreat, take2, headingTags2, feeder


    @staticmethod
    def goToSideB(self, branch, speed, swerve, TrajectoryCommand):
        heading = +60  # side B endpoint is at +60 degrees (West)
        tags = (8, 17)  # side B has AprilTags 8 and 17
        trajectory = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.660, 2.165, heading) if branch == "right" else (3.250, 2.374, heading),
            waypoints=[
                (1.285, 1.135, 54),
            ],
            speed=speed
        )
        return trajectory, (heading, tags)


    @staticmethod
    def goToSideF(self, branch, speed, swerve, TrajectoryCommand):
        heading = -60  # side F endpoint is at -60 degrees (East)
        tags = (6, 19)  # side F has AprilTags 6 and 19
        trajectory = TrajectoryCommand(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.370, 5.646, -60.0) if branch == "right" else (3.650, 5.806, -60.0),
            waypoints=[
                (1.285, 6.915, -54),
            ],
            speed=speed
        )
        return trajectory, (heading, tags)


    @staticmethod
    def approachReef(self, headingTags, height, branch="right", pushFwdSeconds=0.8, speed=1.0, traj=None):
        headingDegrees, tags = headingTags
        assert len(tags) > 0

        # which camera do we use? depends whether we aim for "right" or "left" branch
        assert branch in ("right", "left")
        camera = self.frontLeftCamera if branch == "right" else self.frontRightCamera

        # limelight is slower
        if branch == "left":
            pushFwdSeconds *= 1.3

        from commands.setcamerapipeline import SetCameraPipeline
        from commands.approach import ApproachTag

        approach = ApproachTag(
            camera,
            self.robotDrive,
            headingDegrees,
            speed=speed,
            pushForwardSeconds=pushFwdSeconds,
            dashboardName="auto",
        )

        result = approach.beforeStarting(
            lambda: SmartDashboard.putString("autoStatus", f"apching reef: tags={tags}, h={headingDegrees}")
        )

        result = result.alongWith(AutoFactory.moveArm(self, height=height, final=False))

        # if we have an interruptable trajectory, only run it until `approach` is ready to take over and run
        if traj is not None:
            result = traj.until(lambda: approach.isReady(minRequiredObjectSize=0.8)).andThen(result)

        # if we have specific tags to watch, prepare to watch them
        if tags:
            result = SetCameraPipeline(camera, 0, tags).andThen(result)

        return result


    @staticmethod
    def approachFeeder(self, camera, headingDegrees, speed=1.0, traj=None, tags=None):
        """
        :param traj: trajectory that can be interrupted whenever we can back into the feeder
        :return: a command to back the robot into the feeder
        """
        from commands.approach import ApproachTag

        assert tags, "tags must be specified if you want to back into feeder autonomously"

        if abs(speed) > 1:
            speed = math.copysign(1.0, speed)

        approach = ApproachTag(
            self.rearCamera,
            self.robotDrive,
            headingDegrees,
            speed=speed,
            reverse=True,
            pushForwardSeconds=0.25,  # calibrated for translation gain=0.7
            finalApproachObjSize=2.5,  # calibrated with Eric, Enrique and Davi
            dashboardName="back",
        )

        # 1. the command
        result = approach.beforeStarting(
            lambda: SmartDashboard.putString("autoStatus", f"apching feeder: tags={tags}, h={headingDegrees}")
        )

        # 2. do we have an existing trajectory to terminate when feeder is visible?
        if traj is not None:
            def feederVisible():
                if (
                    approach.isReady()
                    and abs(self.robotDrive.getHeading().degrees()) < 90  # robot is looking at our side of field
                    and self.robotDrive.getPose().x < 4.5  # robot is located between reef and feeder
                ):
                    SmartDashboard.putString(
                        "autoFeederVisible",
                        f"det={camera.hasDetection()}, deg={self.robotDrive.getHeading().degrees()}, x={self.robotDrive.getPose().x}, a={camera.getA()}, delay={camera.getSecondsSinceLastHeartbeat()}"
                    )
                    return True
            result = traj.until(feederVisible).andThen(result)

        # 3. do we have specific tags to watch?
        if tags is not None:
            pipeline = SetCameraPipeline(camera, 0, tags)
            result = pipeline.andThen(result)

        return result


    @staticmethod
    def moveArm(self, height, final=True):
        if TimedCommandRobot.isSimulation():
            return WaitCommand(seconds=1)  # play pretend arm move in simulation

        from commands.elevatorcommands import MoveElevatorAndArm
        from subsystems.arm import ArmConstants

        if height == "intake" or height == "base":
            return MoveElevatorAndArm(self.elevator, 0.0, arm=self.arm, angle=ArmConstants.kArmIntakeAngle)
        if height == "level 2":
            return MoveElevatorAndArm(self.elevator, 4.0, arm=self.arm, angle=ArmConstants.kArmSafeTravelAngle)
        if height == "level 3":
            return MoveElevatorAndArm(self.elevator, 13.0, arm=self.arm, angle=ArmConstants.kArmSafeTravelAngle)
        if height == "level 4":
            angle = ArmConstants.kArmSafeTravelAngle if not final else ArmConstants.kArmLevel4ReleaseAngle
            return MoveElevatorAndArm(self.elevator, 30.0, arm=self.arm, angle=angle)

        assert False, f"height='{height}' is not supported"


    @staticmethod
    def ejectGamepiece(self, calmdownSecondsBeforeFiring=0.0, speed=0.3, timeoutSeconds=0.3):
        from commands.intakecommands import IntakeFeedGamepieceForward
        calmdown = WaitCommand(seconds=calmdownSecondsBeforeFiring)
        shoot = IntakeFeedGamepieceForward(self.intake, speed=speed).withTimeout(timeoutSeconds)
        return calmdown.andThen(shoot)


    @staticmethod
    def intakeGamepiece(self, speed):
        if TimedCommandRobot.isSimulation():
            return WaitCommand(seconds=0.5)  # play pretend, in simulation

        return IntakeGamepiece(self.intake, speed=speed)


    @staticmethod
    def clearDashboard(self):
        fieldDashboard = self.robotDrive.field
        if fieldDashboard is not None:
            fieldDashboard.getObject("start").setPoses([])
            fieldDashboard.getObject("approaching").setPoses([])
            fieldDashboard.getObject("score").setPoses([])
            fieldDashboard.getObject("retreating").setPoses([])
            fieldDashboard.getObject("reload").setPoses([])
            fieldDashboard.getObject("take2").setPoses([])
            fieldDashboard.getObject("score2").setPoses([])


    @staticmethod
    def updateDashboard(self):
        fieldDashboard = self.robotDrive.field
        if fieldDashboard is not None:
            start = self.startPos.getSelected()
            sX, sY, sDeg = start

            goal1branch = self.goal1branch.getSelected()
            goal1traj = self.goal1traj.getSelected()

            headingTags, approach, retreat, take2, headingTags2, feeder = goal1traj(self, start=start, branch=goal1branch)

            heading, tags = headingTags
            heading2, tags2 = headingTags2

            display = lambda t: t.trajectoryToDisplay() if hasattr(t, "trajectoryToDisplay") else []
            approach, retreat, take2 = display(approach), display(retreat), display(take2)

            fieldDashboard.getObject("start").setPoses([Pose2d(Translation2d(sX, sY), Rotation2d.fromDegrees(sDeg))])
            fieldDashboard.getObject("approaching").setPoses(interpolate(approach))
            fieldDashboard.getObject("score").setPoses(scorePoint(approach, heading))
            fieldDashboard.getObject("retreating").setPoses(interpolate(retreat))
            fieldDashboard.getObject("reload").setPoses(scorePoint(retreat[-1:], distance=-0.4))
            fieldDashboard.getObject("take2").setPoses(interpolate(take2))
            fieldDashboard.getObject("score2").setPoses(scorePoint(take2, heading2))


def interpolate(poses, chunks=10):
    result = []
    prev = None
    for pose in poses:
        if prev is None:
            result.append(pose)
        else:
            vector = pose.translation() - prev.translation()
            points = [prev.translation() + (vector * float((1 + chunk) / chunks)) for chunk in range(chunks)]
            result.extend([Pose2d(i, Rotation2d()) for i in points])
        prev = pose
    return result


def scorePoint(approachPoses, headingDegrees=None, distance=0.8):
    if not approachPoses:
        return []
    startPose: Pose2d = approachPoses[-1]
    if headingDegrees is not None:
        heading = Rotation2d.fromDegrees(headingDegrees)
    else:
        heading = startPose.rotation()
    location = startPose.translation() + Translation2d(distance, 0).rotateBy(heading)
    return [Pose2d(location, heading)]


def autoStatus(text) -> Command:
    return InstantCommand(lambda: SmartDashboard.putString("autoStatus", text))


def runCmd(text, command):
    return autoStatus(text).andThen(command)


def backup(pose, factor=0.5):
    x, y, heading = pose
    result = Translation2d(x, y) + Translation2d(-factor * BACKUP_METERS, 0).rotateBy(Rotation2d.fromDegrees(heading))
    return result.x, result.y, heading
