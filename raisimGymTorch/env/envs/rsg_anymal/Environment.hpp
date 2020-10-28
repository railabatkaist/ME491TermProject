//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])

    /// add cylinder
    READ_YAML(double, finalCylinderHeight_, cfg["final_cylinder_height"])
    READ_YAML(double, cylinderRadious_, cfg["cylinder_radious"])
    cylinderHeight_ = finalCylinderHeight_;
    cyilinder_ = world_->addCylinder(cylinderRadious_, cylinderHeight_,9999);
    goalPos_ << 4.0, 0.0, 0.0; posError_.setZero();
    cyilinder_->setPosition(goalPos_(0),goalPos_(1),goalPos_(2) + cylinderHeight_/2);

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(40.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(1.0);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    READ_YAML(int, obDim_, cfg["ob_dim"]);
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_); previousAction_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.6);
    previousAction_ << actionMean_;

    /// Reward coefficients
    READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])
    READ_YAML(double, goalPosRewardCoeff_, cfg["goal_pos_reward_coeff"])

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8080);
      std::cout<<"server on. port: 8080"<<std::endl;
    }
  }

  void init() final {
  }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    torqueReward_ = 0.0;
    goalReward_ = 0.0;

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);

    for(howManySteps_ = 0; howManySteps_ < loopCount; howManySteps_++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
      updateObservation();

      torqueReward_ += torqueRewardCoeff_ * anymal_->getGeneralizedForce().squaredNorm() * getSimulationTimeStep();
      if(posError_.head(2).norm() < 0.05)
        goalReward_ += goalPosRewardCoeff_;

      /// if you want, you can add reward function
      /*************************code************************
       *                                                   *
       *                                                   *
       *                                                   *
       *                                                   *
       *****************************************************/
    }
    previousAction_ << pTarget12_;
    return float(torqueReward_ + goalReward_)/float(howManySteps_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    obDouble_ << gc_(2), /// body height
        baseRot_.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        previousAction_; /// previous action

    /// if you want, you can add observation
    /*************************code************************
     *                                                   *
     *                                                   *
     *                                                   *
     *                                                   *
     *****************************************************/

    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyLinearVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

    controlFrameX_ = {baseRot_[0], baseRot_[1], 0.0};
    controlFrameX_ /= controlFrameX_.norm();
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);
    Eigen::Matrix3d controlRot;
    controlRot << controlFrameX_.e(), controlFrameY_.e(), zAxis_.e();

    posError_ = goalPos_ - gc_.head(3);

    /// if you want add observation, you need to update observation
    /*************************code************************
     *                                                   *
     *                                                   *
     *                                                   *
     *                                                   *
     *****************************************************/
  }

  bool isTerminalState(float& terminalReward) final {

    ///***********************example*********************
    /// if the self collision -> terminal
    for(auto& contact: anymal_->getContacts())
      if(contact.isSelfCollision())
        return true;
    ///***********************example*********************

    /// if you want, you can use another terminal condition
    /*************************code************************
     *                                                   *
     *                                                   *
     *                                                   *
     *                                                   *
     *****************************************************/
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() final {
    ///***********************example*********************
    /// curriculum factor increase 0.07->1 slowly
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    ///***********************example*********************

    /// if you want, you can add curriculum update
    /*************************code************************
     *                                                   *
     *                                                   *
     *                                                   *
     *                                                   *
     *****************************************************/
    RSINFO_IF(visualizable_, "Curriculum factor: "<<curriculumFactor_)
  }

  void getConstraints(Eigen::Ref<EigenVec> constraints){

  }

 private:
  int gcDim_, gvDim_, nJoints_;
  int howManySteps_ = 0;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::Cylinder* cyilinder_;
  raisim::Mat<3,3> baseRot_;
  raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;

  double terminalRewardCoeff_ = -10.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double goalOriRewardCoeff_ = 0.0, goalPosRewardCoeff_ = 0.0, goalVelRewardCoeff_ = 0.0, goalReward_ = 0.0;

  double curriculumFactor_, curriculumDecayFactor_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, previousAction_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  Eigen::Vector3d goalPos_, posError_, goalInControlFrame_;
  Eigen::Vector3d baseTogoalDir_;
  double cylinderRadious_, cylinderHeight_ , finalCylinderHeight_;
};
}

