import torch
from torch import nn
import os
from tqdm import tqdm

def train(
        model, optimizer, scheduler, fastspeech_loss,
        train_config, training_loader, logger
    ):

    current_step = 0
    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)

                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)

                duration = db["duration"].int().to(train_config.device)
                energy = db["energy"].float().to(train_config.device)
                pitch = db["pitch"].float().to(train_config.device)

                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                (
                    mel_output,
                    log_duration_prediction,
                    energy_prediction,
                    pitch_prediction
                ) = model(
                    character, src_pos, mel_pos,
                    energy, pitch, duration,
                    mel_max_length=max_mel_len,
                )

                # Calc Loss
                mel_loss, d_loss, e_loss, p_loss = fastspeech_loss(
                    mel_output, mel_target,
                    log_duration_prediction, duration,
                    energy_prediction, energy,
                    pitch_prediction, pitch
                )
                total_loss = mel_loss + d_loss + e_loss + p_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = d_loss.detach().cpu().numpy()
                e_l = e_loss.detach().cpu().numpy()
                p_l = p_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("energy_loss", e_l)
                logger.add_scalar("pitch_loss", p_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_config': model.model_config,
                        'mel_config': model.mel_config,
                        'train_config': train_config
                    }, os.path.join(train_config.save_dir, 'checkpoint_%d.pth.tar' % current_step))

                    print("save model at step %d ..." % current_step)



