#!/bin/bash
# aarch64-linux-gnu-ar -x libopencv_dnn.a
# aarch64-linux-gnu-ar -x libopencv_ml.a
# aarch64-linux-gnu-ar -x libopencv_objdetect.a
# aarch64-linux-gnu-ar -x libopencv_shape.a
# aarch64-linux-gnu-ar -x libopencv_stitching.a
# aarch64-linux-gnu-ar -x libopencv_superres.a
# aarch64-linux-gnu-ar -x libopencv_videostab.a
# aarch64-linux-gnu-ar -x libopencv_calib3d.a
# aarch64-linux-gnu-ar -x libopencv_features2d.a
# aarch64-linux-gnu-ar -x libopencv_highgui.a
# aarch64-linux-gnu-ar -x libopencv_videoio.a
# aarch64-linux-gnu-ar -x libopencv_imgcodecs.a
# aarch64-linux-gnu-ar -x libopencv_photo.a
# aarch64-linux-gnu-ar -x libopencv_imgproc.a
# aarch64-linux-gnu-ar -x libopencv_flann.a
# aarch64-linux-gnu-ar -x libopencv_core.a

# aarch64-linux-gnu-ar -x liblibprotobuf.a
# aarch64-linux-gnu-ar -x libzlib.a
# aarch64-linux-gnu-ar -x liblibjpeg-turbo.a
# aarch64-linux-gnu-ar -x liblibwebp.a
# aarch64-linux-gnu-ar -x liblibpng.a
# aarch64-linux-gnu-ar -x liblibtiff.a
# aarch64-linux-gnu-ar -x liblibjasper.a
# aarch64-linux-gnu-ar -x libIlmImf.a
# aarch64-linux-gnu-ar -x libquirc.a
# aarch64-linux-gnu-ar -x libtegra_hal.a
# aarch64-linux-gnu-ar -x libeasydk.a
# aarch64-linux-gnu-ar -x libturbojpeg.a
# aarch64-linux-gnu-ar -x libjpeg.a
# aarch64-linux-gnu-ar -x libyuv.a

aarch64-linux-gnu-ar csrT libopencv.a libopencv_dnn.a \
    libopencv_ml.a libopencv_objdetect.a libopencv_shape.a libopencv_stitching.a \
    libopencv_superres.a libopencv_videostab.a libopencv_calib3d.a libopencv_features2d.a \
    libopencv_highgui.a libopencv_videoio.a libopencv_imgcodecs.a libopencv_photo.a libopencv_imgproc.a \
    libopencv_flann.a libopencv_core.a liblibprotobuf.a libzlib.a liblibjpeg-turbo.a liblibwebp.a \
    liblibpng.a  liblibtiff.a liblibjasper.a libIlmImf.a libquirc.a libtegra_hal.a libeasydk.a 
# aarch64-linux-gnu-ar -rc libopencv_easydk.a *.o
# rm *.o

