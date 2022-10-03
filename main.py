import cv2 
import os
import shutil
import numpy as np


def create_entry_folder():
  if "entry_frames" not in os.listdir():
    os.mkdir("entry_frames")
  else:
    shutil.rmtree("entry_frames")
    os.mkdir("entry_frames")


def create_exit_folder():
  if "exit_frames" not in os.listdir():
    os.mkdir("exit_frames")
  else:
    shutil.rmtree("exit_frames")
    os.mkdir("exit_frames")


def separate_video_into_frames(video_name):
  vidcap = cv2.VideoCapture(video_name)
  success, image = vidcap.read()
  count = 1

  while success:
    cv2.imwrite(f"entry_frames/{count}.jpg", image)     # save frame as JPEG file      
    success, image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1


def create_output_video(frames, file_name = "project.mp4"):
    frame = frames[0]
    num_of_rows, num_of_columns, _ = frame.shape
    resolution = (num_of_columns, num_of_rows)
    
    video_out = cv2.VideoWriter(f'videos/{file_name}',cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    
    for i in range(len(frames)):
        frame = frames[i]
        cv2.imwrite(f"exit_frames/{i+1}.jpg", frame)
        video_out.write(frame)
    
    video_out.release()
    


def order_frames(frame):
    numeration = int(frame.split(".")[0])
    return numeration


def color_segmentation_rgb(frames, r_interval: tuple, g_interval: tuple, b_interval: tuple, segment_color: tuple):
  for i in range(len(frames)):
      frame = frames[i]
      num_of_rows, num_of_columns, _ = frame.shape
      for row in range(num_of_rows):
          for column in range(num_of_columns):
              b, g, r = frame.item(row, column, 0), frame.item(row, column, 1), frame.item(row, column, 2)
              if (r_interval[0] <= r <= r_interval[1] and g_interval[0] <= g <= g_interval[1] and b_interval[0] <= b <= b_interval[1]):
                  frame.itemset((row, column, 0), segment_color[0])
                  frame.itemset((row, column, 1), segment_color[1])
                  frame.itemset((row, column, 2), segment_color[2])
              else:
                  frame.itemset((row, column, 0), 0)
                  frame.itemset((row, column, 1), 0)
                  frame.itemset((row, column, 2), 0)
  return frames


# def color_segmentation_hsv(frames, h_interval: tuple, s_interval: tuple, v_interval: tuple, segment_color: tuple):
#   for i in range(len(frames)):
#       frame_hsv =  cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
#       num_of_rows, num_of_columns, _ = frame_hsv.shape
#       for row in range(num_of_rows):
#           for column in range(num_of_columns):
#               h, s, v = frame_hsv.item(row, column, 0), frame_hsv.item(row, column, 1), frame_hsv.item(row, column, 2)
#               if (h_interval[0] <= h <= h_interval[1] and s_interval[0] <= s <= s_interval[1] and v_interval[0] <= v <= v_interval[1]):
#                   frame_hsv.itemset((row, column, 0), segment_color[0])
#                   frame_hsv.itemset((row, column, 1), segment_color[1])
#                   frame_hsv.itemset((row, column, 2), segment_color[2])
#               else:
#                   frame_hsv.itemset((row, column, 0), 0)
#                   frame_hsv.itemset((row, column, 1), 0)
#                   frame_hsv.itemset((row, column, 2), 0)

#       frames[i] = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
#   return frames


def color_segmentation_hsv(frames, h_interval: tuple, s_interval: tuple, v_interval: tuple, segment_color: tuple):
  for i in range(len(frames)):
      frame_hsv =  cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
      segment_min = np.array([h_interval[0], s_interval[0], v_interval[0]], np.uint8)
      segment_max = np.array([h_interval[1], s_interval[1], v_interval[1]], np.uint8)

      frame_hsv = cv2.inRange(frame_hsv, segment_min, segment_max)
      frames[i] = cv2.cvtColor(frame_hsv, cv2.COLOR_GRAY2BGR)
  return frames


def bin_frames_dilation(frames, kernel_size: tuple, iterations: int):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
  for i in range(len(frames)):
      frame = frames[i]
      frame = cv2.dilate(frame, kernel, iterations=iterations)
      frames[i] = frame
  return frames


def bin_frames_erosion(frames, kernel_size: tuple, iterations: int):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
  for i in range(len(frames)):
      frame = frames[i]
      frame = cv2.erode(frame, kernel, iterations=iterations)
      frames[i] = frame
  return frames


def bin_frame_dilation(frame, kernel_size: tuple, iterations: int):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
  frame = cv2.dilate(frame, kernel, iterations=iterations)
  return frame


def bin_frames_closing(frames, kernel_size: tuple, iterations: int):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
  for i in range(len(frames)):
      frame = frames[i]
      frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations)
      frames[i] = frame
  return frames


def bin_frames_opening(frames, kernel_size: tuple, iterations: int):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
  for i in range(len(frames)):
      frame = frames[i]
      frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=iterations)
      frames[i] = frame
  return frames


def frames_gaussian_blur(frames, kernel_size: tuple):
  for i in range(len(frames)):
      frame = frames[i]
      frame = cv2.GaussianBlur(frame, (kernel_size[0], kernel_size[1]), 0)
      frames[i] = frame
  return frames


def create_light_saber(original_frames, light_saber_frames, saber_edge_color: tuple = (0, 255, 0), saber_center_color: tuple = (255, 255, 255)):
  result_frames = []
  for i in range(len(original_frames)):
    result_frame = original_frames[i].copy()
    result_frames.append(result_frame)
    light_saber_frame = light_saber_frames[i]
    num_of_rows, num_of_columns, _ = light_saber_frame.shape
    for row in range(num_of_rows):
      for column in range(num_of_columns):
        if light_saber_frame.item(row, column, 0) != 0:
          result_frame.itemset((row, column, 0), saber_center_color[0])
          result_frame.itemset((row, column, 1), saber_center_color[1])
          result_frame.itemset((row, column, 2), saber_center_color[2])
  
  frames_gaussian_blur(light_saber_frames, (29, 29))
  for i in range(len(original_frames)):
    result_frame = result_frames[i]
    original_frame = original_frames[i]
    light_saber_frame = light_saber_frames[i]
    num_of_rows, num_of_columns, _ = light_saber_frame.shape
    for row in range(num_of_rows):
      for column in range(num_of_columns):
        if result_frame.item(row, column, 0) == saber_center_color[0] and result_frame.item(row, column, 1) == saber_center_color[1] and result_frame.item(row, column, 2) == saber_center_color[2]:
          smoothing_factor = light_saber_frame.item(row, column, 0) / 255
          smoothing_factor -= 0.4
          b, g, r = original_frame.item(row, column, 0), original_frame.item(row, column, 1), original_frame.item(row, column, 2)
          if light_saber_frame.item(row, column, 0) < saber_center_color[0] and light_saber_frame.item(row, column, 0) > 0:
            result_frame.itemset((row, column, 0), saber_edge_color[0] * smoothing_factor + b * (1 - smoothing_factor))
            result_frame.itemset((row, column, 1), saber_edge_color[1] * smoothing_factor + g * (1 - smoothing_factor))
            result_frame.itemset((row, column, 2), saber_edge_color[2] * smoothing_factor + r * (1 - smoothing_factor))

  return result_frames


def main():
    nome_do_arquivo = "videos/"
    # nome_do_arquivo += "sabre_ciano_def_480p_3seg.mp4"
    # nome_do_arquivo += "sabre_ciano_480p.mp4"
    nome_do_arquivo += "entry.mp4"
    create_entry_folder()
    create_exit_folder()
    separate_video_into_frames(nome_do_arquivo)

    frames_str = [frame for frame in os.listdir("entry_frames")]
    frames_str.sort(key = order_frames)
    original_frames = []
    frames = []
    for frame_str in frames_str:
        original_frames.append(cv2.imread("entry_frames/" + frame_str))
        frames.append(cv2.imread("entry_frames/" + frame_str))
    
    # color_segmentation(frames, (0, 80), (0, 200), (100, 255), (255, 255, 255))
    # color_segmentation_rgb(frames, (0, 70), (100, 150), (100, 150), (255, 255, 255))
    # color_segmentation_hsv(frames, (170, 180), (0, 255), (0, 255), (0, 0, 255))
    color_segmentation_hsv(frames, (86, 100), (50, 240), (45, 255), (0, 0, 255))
    bin_frames_opening(frames, (3, 3), 1)
    bin_frames_closing(frames, (20, 20), 1)
    bin_frames_dilation(frames, (40, 40), 1)

    frames = create_light_saber(original_frames, frames, (237, 187, 99), (255, 255, 255))

    create_output_video(frames)


if __name__ == "__main__":
    main()
