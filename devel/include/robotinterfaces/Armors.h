// Generated by gencpp from file robotinterfaces/Armors.msg
// DO NOT EDIT!


#ifndef ROBOTINTERFACES_MESSAGE_ARMORS_H
#define ROBOTINTERFACES_MESSAGE_ARMORS_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <robotinterfaces/Armor.h>

namespace robotinterfaces
{
template <class ContainerAllocator>
struct Armors_
{
  typedef Armors_<ContainerAllocator> Type;

  Armors_()
    : header()
    , armors()  {
    }
  Armors_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , armors(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector< ::robotinterfaces::Armor_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::robotinterfaces::Armor_<ContainerAllocator> >> _armors_type;
  _armors_type armors;





  typedef boost::shared_ptr< ::robotinterfaces::Armors_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::robotinterfaces::Armors_<ContainerAllocator> const> ConstPtr;

}; // struct Armors_

typedef ::robotinterfaces::Armors_<std::allocator<void> > Armors;

typedef boost::shared_ptr< ::robotinterfaces::Armors > ArmorsPtr;
typedef boost::shared_ptr< ::robotinterfaces::Armors const> ArmorsConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::robotinterfaces::Armors_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::robotinterfaces::Armors_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::robotinterfaces::Armors_<ContainerAllocator1> & lhs, const ::robotinterfaces::Armors_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.armors == rhs.armors;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::robotinterfaces::Armors_<ContainerAllocator1> & lhs, const ::robotinterfaces::Armors_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace robotinterfaces

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::robotinterfaces::Armors_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robotinterfaces::Armors_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robotinterfaces::Armors_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robotinterfaces::Armors_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robotinterfaces::Armors_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robotinterfaces::Armors_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::robotinterfaces::Armors_<ContainerAllocator> >
{
  static const char* value()
  {
    return "69c17fbb4896cb79d8e99efc826b5907";
  }

  static const char* value(const ::robotinterfaces::Armors_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x69c17fbb4896cb79ULL;
  static const uint64_t static_value2 = 0xd8e99efc826b5907ULL;
};

template<class ContainerAllocator>
struct DataType< ::robotinterfaces::Armors_<ContainerAllocator> >
{
  static const char* value()
  {
    return "robotinterfaces/Armors";
  }

  static const char* value(const ::robotinterfaces::Armors_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::robotinterfaces::Armors_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n"
"Armor[] armors\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: robotinterfaces/Armor\n"
"string number\n"
"string type\n"
"float32 distance_to_image_center\n"
"geometry_msgs/Pose pose\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::robotinterfaces::Armors_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::robotinterfaces::Armors_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.armors);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Armors_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::robotinterfaces::Armors_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::robotinterfaces::Armors_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "armors[]" << std::endl;
    for (size_t i = 0; i < v.armors.size(); ++i)
    {
      s << indent << "  armors[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::robotinterfaces::Armor_<ContainerAllocator> >::stream(s, indent + "    ", v.armors[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROBOTINTERFACES_MESSAGE_ARMORS_H
