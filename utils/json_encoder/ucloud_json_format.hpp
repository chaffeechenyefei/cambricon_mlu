#if ( !defined(_UCLOUD_JSON_FORMAT_HPP_) || defined(GENERATE_ENUM_STRINGS) )

#if (!defined(GENERATE_ENUM_STRINGS))
	#define _UCLOUD_JSON_FORMAT_HPP_
#endif

#include "EnumToString.hpp"
BEGIN_ENUM(JSON_ROOT)
{
    DECL_ENUM_ELEMENT(FACE_ATTRIBUTION),
    DECL_ENUM_ELEMENT(VEHICLE),
    DECL_ENUM_ELEMENT(OTHERS),
}
END_ENUM(JSON_ROOT)

BEGIN_ENUM(JSON_ATTR)
{
    DECL_ENUM_ELEMENT(AGE),
    DECL_ENUM_ELEMENT(SEX),
    DECL_ENUM_ELEMENT(MASK),
    DECL_ENUM_ELEMENT(LICPLATE),
    DECL_ENUM_ELEMENT(NOTE),
    DECL_ENUM_ELEMENT(OTHERS),
}
END_ENUM(JSON_ATTR)

#endif // (!defined(DAYS_H) || defined(GENERATE_ENUM_STRINGS))
