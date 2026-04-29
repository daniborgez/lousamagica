import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

def main():
    # Inicializa a câmera e define a resolução
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Largura
    cap.set(4, 720)  # Altura

    # Inicializa o detector de mãos (confiança de 80%, rastreia 1 mão no máximo)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # Cria o "Canvas" (a tela transparente onde o desenho vai ficar armazenado)
    # Tem que ter o mesmo tamanho da resolução da câmera
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Posições anteriores de X e Y
    xp, yp = 0, 0

    print("--- Lousa Mágica (Hand Tracking) Iniciada ---")
    print("👆 Apenas Indicador levantado: DESENHA")
    print("✌️ Indicador e Médio levantados: MOVE (sem desenhar)")
    print("⌨️ Pressione 'c' para LIMPAR a lousa.")
    print("⌨️ Pressione 'q' para SAIR.")

    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Inverte a imagem para efeito espelho (mais natural)
        img = cv2.flip(img, 1)
        
        # Encontra as mãos na imagem. O draw=False tira o esqueleto padrão, mas vamos deixar True para você ver funcionando.
        hands, img = detector.findHands(img, draw=True)

        if hands:
            hand = hands[0] # Pega a primeira mão detectada
            lmList = hand["lmList"] # Lista de 21 pontos (Landmarks) da mão
            
            if len(lmList) != 0:
                # O ponto 8 é a ponta do dedo indicador
                x1, y1, _ = lmList[8]
                # O ponto 12 é a ponta do dedo médio
                x2, y2, _ = lmList[12]
                
                # Verifica quais dedos estão levantados
                # Retorna uma lista ex: [0, 1, 0, 0, 0] (Apenas indicador levantado)
                fingers = detector.fingersUp(hand)

                # 1. Modo de Movimento/Hover: Indicador e Médio levantados
                if fingers[1] == 1 and fingers[2] == 1:
                    xp, yp = 0, 0 # Reseta a posição anterior para a linha não "pular"
                    # Desenha um círculo na ponta do dedo para indicar que está selecionando
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                # 2. Modo de Desenho: Apenas o indicador levantado
                if fingers[1] == 1 and fingers[2] == 0:
                    # Círculo verde para indicar que está desenhando
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    
                    # Se for o primeiro frame desenhando, iguala a posição atual à passada
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                        
                    # Desenha a linha no nosso Canvas
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 255, 0), 10) # Linha Verde, espessura 10
                    
                    # Atualiza a posição
                    xp, yp = x1, y1

        # MÁGICA DA SOBREPOSIÇÃO: Colocando o canvas desenhado em cima do vídeo ao vivo
        # 1. Converte o canvas para cinza
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        # 2. Cria uma máscara invertida (onde tem desenho fica preto, resto fica branco)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        # 3. Faz um "buraco" no frame original no exato local do desenho
        img = cv2.bitwise_and(img, imgInv)
        # 4. Preenche esse buraco com as cores do Canvas
        img = cv2.bitwise_or(img, imgCanvas)

        # Mostra o resultado final
        cv2.imshow("Lousa Magica - Hand Tracking", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Zera a matriz do canvas, limpando a tela
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()