import tensorflow as tf
from nets.yolo import get_yolo_loss
from tqdm import tqdm


# Prevent bugs
def get_train_step_fn():
    @tf.function
    def train_step(imgs, targets, net, yolo_loss, optimizer):
        with tf.GradientTape() as tape:
            # Calculate loss
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + [targets]
            
            loss_value  = yolo_loss(args)
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def val_step(imgs, targets, net, yolo_loss, optimizer):
    # Calculate
    P5_output, P4_output, P3_output = net(imgs, training=False)
    args        = [P5_output, P4_output, P3_output] + [targets]
    loss_value  = yolo_loss(args)
    loss_value  = tf.reduce_sum(net.losses) + loss_value
    return loss_value

def fit_one_epoch(net, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, num_classes):
    train_step  = get_train_step_fn()
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            targets     = tf.convert_to_tensor(targets)
            loss_value  = train_step(images, targets, net, yolo_loss, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            targets     = tf.convert_to_tensor(targets)
            loss_value  = val_step(images, targets, net, yolo_loss, optimizer)
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
