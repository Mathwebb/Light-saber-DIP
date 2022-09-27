import cv2 
import os
import numpy as np
import glob

def criar_pasta_de_entrada():
  if "entry_frames" not in os.listdir():
    os.mkdir("entry_frames")

def separar_video_em_frames(nome_do_video):
  vidcap = cv2.VideoCapture(nome_do_video)
  success,image = vidcap.read()
  count = 1

  while success:
    cv2.imwrite(f"entry_frames/{count}.jpg", image)     # save frame as JPEG file      
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1

def ordenar_frames(frame):
  numeracao = int(frame.split(".")[0])
  return numeracao

def criar_pasta_de_saida():
  if "exit_frames" not in os.listdir():
    os.mkdir("exit_frames")

def processar_frames(frames):
  for frame in frames:
    img = cv2.imread("entry_frames/" + frame)
    num_de_linhas, num_de_colunas, _ = img.shape
    img2 = np.zeros((img.shape), np.uint8)

    for i in range(num_de_linhas):
      for j in range(num_de_colunas):
        pixel_b = img.item(i, j, 0)
        pixel_g = img.item(i, j, 1)
        pixel_r = img.item(i, j, 2)

        if 45 <= pixel_b <= 255 and 0 <= pixel_g <= 120 and 0 <= pixel_r <= 20:
          img2.itemset((i, j, 0), 0)
          img2.itemset((i, j, 1), 255)
          img2.itemset((i, j, 2), 0)
        
        else:
          img2.itemset((i, j, 0), pixel_b)    
          img2.itemset((i, j, 1), pixel_g)
          img2.itemset((i, j, 2), pixel_r)

    cv2.imwrite(f"exit_frames/{frame}", img2)

def criar_video(frames):
  frame = frames[0]
  img = cv2.imread(f"exit_frames/{frame}")
  num_de_linhas, num_de_colunas, _ = img.shape
  resolucao = (num_de_colunas, num_de_linhas)

  out = cv2.VideoWriter('videos/project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, resolucao)
  
  for frame in frames:
    img = cv2.imread(f"exit_frames/{frame}")
    out.write(img)

  out.release()

def main():
  criar_pasta_de_entrada()
  separar_video_em_frames("videos/cadeira.mp4")
  frames = [frame for frame in os.listdir("entry_frames")]
  criar_pasta_de_saida()
  processar_frames(frames)
  frames_de_saida = [frame for frame in os.listdir("exit_frames")]
  frames_de_saida.sort(key = ordenar_frames)
  criar_video(frames_de_saida)

  img = cv2.imread("frames/1.jpg")
  num_de_linhas, num_de_colunas, _ = img.shape
  img2 = np.zeros((img.shape), np.uint8)

  for i in range(num_de_linhas):
    for j in range(num_de_colunas):
      pixel_b = img.item(i, j, 0)
      pixel_g = img.item(i, j, 1)
      pixel_r = img.item(i, j, 2)

      if 45 <= pixel_b <= 255 and 0 <= pixel_g <= 120 and 0 <= pixel_r <= 20:
        img2.itemset((i, j, 0), 255)
        img2.itemset((i, j, 1), 255)
        img2.itemset((i, j, 2), 255)
      
      else:
        img2.itemset((i, j, 0), pixel_b)    
        img2.itemset((i, j, 1), pixel_g)
        img2.itemset((i, j, 2), pixel_r)

  cv2.imwrite("1.jpg", img2)


if __name__ == "__main__":
  main()