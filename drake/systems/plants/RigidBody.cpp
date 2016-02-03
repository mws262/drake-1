
#include "RigidBody.h"
#include <stdexcept>
#include <numeric>

using namespace std;
using namespace Eigen;


RigidBody::RigidBody() :
    parent(nullptr),
    collision_filter_group(DrakeCollision::DEFAULT_GROUP),
    collision_filter_ignores(DrakeCollision::NONE_MASK)
{
  robotnum = 0;
	position_num_start = 0;
	velocity_num_start = 0;
	body_index = 0;
	mass = 0.0;
  com = Vector3d::Zero();
	I << Matrix<double, TWIST_SIZE, TWIST_SIZE>::Zero();
}

void RigidBody::setJoint(std::unique_ptr<DrakeJoint> new_joint)
{
  this->joint = move(new_joint);
}

const DrakeJoint& RigidBody::getJoint() const
{
  if (joint) {
    return (*joint);
  }
  else {
    throw runtime_error("Joint is not initialized");
  }
}

bool RigidBody::hasParent() const {
  return parent !=nullptr;
}


void RigidBody::addVisualElement(const DrakeShapes::VisualElement& element)
{
  visual_elements.push_back(element);
}

const DrakeShapes::VectorOfVisualElements& RigidBody::getVisualElements() const
{
  return visual_elements;
}

void RigidBody::setCollisionFilter(const DrakeCollision::bitmask& group, 
                                   const DrakeCollision::bitmask& ignores)
{
  setCollisionFilterGroup(group);
  setCollisionFilterIgnores(ignores);
}

bool RigidBody::appendCollisionElementIdsFromThisBody(const string& group_name, vector<DrakeCollision::ElementId>& ids) const
{
  auto group_ids_iter = collision_element_groups.find(group_name);
  if (group_ids_iter != collision_element_groups.end()) {
    ids.reserve(ids.size() + distance(group_ids_iter->second.begin(), group_ids_iter->second.end()));
    ids.insert(ids.end(), group_ids_iter->second.begin(), group_ids_iter->second.end());
    return true;
  } else {
    return false;
  }
}

bool RigidBody::appendCollisionElementIdsFromThisBody(vector<DrakeCollision::ElementId>& ids) const
{
  auto add_element_count = [](size_t count, const pair<string, vector<DrakeCollision::ElementId>>& group) {return count + group.second.size(); };
  size_t num_elements = accumulate(collision_element_groups.begin(), collision_element_groups.end(), size_t(0), add_element_count);

  ids.reserve(ids.size() + num_elements);
  for (const auto& group : collision_element_groups) {
    ids.insert(ids.end(), group.second.begin(), group.second.end());
  }
  return true;
}

RigidBody::CollisionElement::
CollisionElement( const CollisionElement& other)
  : DrakeCollision::Element(other), body(other.getBody()) {}

  RigidBody::CollisionElement::
CollisionElement( const Isometry3d& T_element_to_link, std::shared_ptr<RigidBody> body)
  : DrakeCollision::Element(T_element_to_link), body(body) {}

  RigidBody::CollisionElement::
CollisionElement(const DrakeShapes::Geometry& geometry,
    const Isometry3d& T_element_to_link, std::shared_ptr<RigidBody> body)
  : DrakeCollision::Element(geometry, T_element_to_link), body(body) {}

RigidBody::CollisionElement* RigidBody::CollisionElement::clone() const
{
  return new CollisionElement(*this);
}

const std::shared_ptr<RigidBody>& RigidBody::CollisionElement:: getBody() const
{
  return this->body;
}

bool RigidBody::CollisionElement::collidesWith( const DrakeCollision::Element* other) const
{
  //DEBUG
  //cout << "RigidBody::CollisionElement::collidesWith: START" << endl;
  //END_DEBUG
  auto other_rb = dynamic_cast<const RigidBody::CollisionElement*>(other);
  bool collides = true;
  if (other_rb != nullptr) {
    collides = this->body->collidesWith(other_rb->body);
    //DEBUG
    //cout << "RigidBody::CollisionElement::collidesWith:" << endl;
    //cout << "  " << this->body->linkname << " & " << other_rb->body->linkname;
    //cout << ": collides = " << collides << endl;
    //END_DEBUG
  }   
  return collides;
}

ostream &operator<<( ostream &out, const RigidBody &b)
{
  std::string joint_name = b.hasParent() ? b.getJoint().getName() : "no parent joint";
  out << "RigidBody(" << b.linkname << "," << joint_name << ")";
  return out;
}
