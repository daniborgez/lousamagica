import cv2
import numpy as np

def main():
    # Inicializa a câmera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Cria o "Canvas" (a tela em branco para armazenar o desenho)
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Posições anteriores
    prev_x, prev_y = 0, 0

    print("--- Lousa Mágica (Light Tracking) ---")
    print("🔦 Aponte a LANTERNA do celular para a câmera para desenhar!")
    print("🖐️ Esconda a luz com a mão para parar de desenhar.")
    print("⌨️ Pressione 'c' para limpar a tela.")
    print("⌨️ Pressione 'q' para sair.")

    while True:
        success, img = cap.read()
        if not success:
            break

        # Inverte para efeito espelho
        img = cv2.flip(img, 1)

        # 1. Transforma a imagem em tons de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplica um desfoque forte para ignorar pequenos reflexos e focar só na luz principal
        gray = cv2.GaussianBlur(gray, (25, 25), 0)

        # 3. O OpenCV acha matematicamente as coordenadas exatas do ponto mais claro da tela
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # Se o brilho for muito alto (próximo a 255, que é o branco puro), sabemos que é a lanterna
        if maxVal > 240:
            cx, cy = maxLoc # Coordenadas X e Y da lanterna
            
            # Desenha um círculo vermelho na ponta da luz para você ver que foi detectado
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)

            # Se for o primeiro momento que a luz apareceu, igualamos os pontos
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Desenha a linha verde no canvas
            cv2.line(imgCanvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 10)

            # Atualiza a posição
            prev_x, prev_y = cx, cy
        else:
            # Se você esconder a luz, ele reseta a posição (para a linha não pular pela tela)
            prev_x, prev_y = 0, 0

        # --- MÁGICA DA SOBREPOSIÇÃO (Mesmo esquema de antes) ---
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        # --------------------------------------------------------

        cv2.imshow("Lousa Magica - Lanterna", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()