import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training = False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class NetworkConstructor:
    def __init__(self):
        pass
    def addConv(self, input_layer, filter_shape, bn = True, activate = True, downsample = False):
        # filter_shape is [H, W, IC, OC]
        if downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            strides = 2
            padding = 'valid'
        else:
            strides = 1
            padding = 'same'
        active_type = "leaky"
        assert filter_shape[0] == filter_shape[1]
        conv = tf.keras.layers.Conv2D(
                filters             = filter_shape[-1],
                kernel_size         = filter_shape[0],
                strides             = strides,
                padding             = padding,
                use_bias            = not bn,
                kernel_regularizer  = tf.keras.regularizers.l2(0.0005),
                kernel_initializer  = tf.random_normal_initializer(stddev=0.01),
                bias_initializer    = tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate:
            assert active_type == "leaky"
            conv = tf.nn.leaky_relu(conv, alpha = 0.1)
        return conv
    def addMaxpool(self, input_layer, dim):
        conv = tf.keras.layers.MaxPool2D(dim, dim, 'same')(input_layer)
        return conv
    def addUpsample(self, input_layer):
        upsampled = tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
        return unpsampled
    def addUpsampleSpecific(self, output_size, input_layer):
        upsampled = tf.image.resize(input_layer, output_size, method='bilinear')
        return upsampled
    def buildDarknet53Tiny(self, input_layer):
        conv = self.addConv(input_layer, (3, 3, 3, 16))
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 16, 32))
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 32, 64))
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 64, 128))
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 128, 256))
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 256, 512))
        route_1 = conv
        conv = self.addMaxpool(conv, 2)
        conv = self.addConv(conv, (3, 3, 512, 1024))
        return route_1, conv

    def buildPAN(self, input0, input1):
        route = input1
        conv = self.addConv(input1, (1, 1, 512, 256))
        conv = self.addUpsampleSpecific((13, 13), conv)
        # (13, 13, 256) (13, 13, 256)
        conv = tf.concat([input0, conv], axis = -1)
        return input1, conv

    def buildYolov3(self, input0, input1, num_class):
        conv = self.addConv(input1, (3, 3, 768, 1024))
        conv = self.addConv(conv, (1, 1, 1024, 512))
        route = conv
        conv_mbox = self.addConv(conv, (1, 1, 512, 3 * (num_class + 5)), activate = False, bn = False)
        conv = self.addConv(route, (3, 3, 512, 1024), downsample = True)
        conv = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(conv)
        conv = tf.concat([input0, conv], axis = -1)
        conv_lbox = self.addConv(conv, (1, 1, 2048, 3 * (num_class + 5)), activate = False, bn = False)
        return conv_mbox, conv_lbox

    def buildDecoder(self, conv, size, num_class, i, strides, anchors, xyscale):
        conv_output = tf.reshape(conv, 
                (tf.shape(conv)[0], size, size, 3, 5 + num_class))
        print(conv)
        print(conv_output)
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, num_class), axis = -1)
        print(conv_raw_dxdy)
        print(conv_raw_dwdh)
        print(conv_raw_conf)
        print(conv_raw_prob)
        xy_grid = tf.meshgrid(tf.range(size), tf.range(size))
        # after tf.stack, there will be (size, size, 2) (coords)
        # after expand_dims, there will be (size, size, 1, 2)
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis = -1), axis = 2)
        # tf.tile([1, size, size, 1, 2], [None, 1, 1, 3, 1])
        # xy_grid = [1, size, size, 3, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis = 0), [tf.shape(conv)[0], 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
        print(conv_raw_dxdy)
        print(xy_grid)
        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * xyscale[i]) - 0.5 * (xyscale[i] - 1) + xy_grid) * strides[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis = -1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)
    
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis = -1)


if __name__ == '__main__':
    ih = 416
    iw = 416
    ic = 3

    network_constructor = NetworkConstructor()

    # human 
    num_class = 1;

    # Network building
    # Input shuold have input dimension except batch size
    input_layer = tf.keras.layers.Input([ih, iw, ic])
    # Yolov4 is composed of backbone, neck and head
    # Backbone will be a resnet (due to implementation complexity) and SPP (dakrnet53_tiny)
    # Neck will be Path aggregation network with upsampling
    # Head will be a 1 stair(?) yolov3 (only have one freeze layer)

    # Backbone build
    route_1, conv = network_constructor.buildDarknet53Tiny(input_layer)
    # Neck build
    route_1, conv = network_constructor.buildPAN(route_1, conv)
    # Head build
    conv_mbox, conv_lbox = network_constructor.buildYolov3(route_1, conv, num_class)
    print(conv_mbox, conv_lbox)

    bbox_tensors = []
    # Yolo layer to decode conv_m/lbox
    bbox_tensor = network_constructor.buildDecoder(conv_mbox, 13, num_class, 0, [8, 16], [12, 16], [1.2, 1.1])
    bbox_tensors.append(conv_mbox)
    bbox_tensors.append(bbox_tensor)
    bbox_tensor = network_constructor.buildDecoder(conv_lbox, 13, num_class, 1, [8, 16], [12, 16], [1.2, 1.1])
    bbox_tensors.append(conv_lbox)
    bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        # TODO: lossfunction calculation add
        pass
