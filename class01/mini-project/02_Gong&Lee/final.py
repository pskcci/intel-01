from faceChecking import faceCheckings
from dual import duals
from dual import save_dataset
import cv2
import time


def resize_image(image, fixed_size=(600, 600)):
    return cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    while True:
        total_result = None
        # result = 0
        # if result == "{'rps_result': [0]}":
        #     result = 1
        # elif result == "{'rps_result': [5]}":
        #     result = 2
        # elif result == "{'rps_result': [9]}":
        #     result = 3
        # else:
        #     print("None type data check plz")
        value1 = []
        value2 = []

        # Player1
        player1 = faceCheckings()
        result = duals()
        if result and 'rps_result' in result:
            rps_result_list = result['rps_result']
        if rps_result_list == [0]:
            value1 = 1
        elif rps_result_list == [5]:
            value1 = 2
        elif rps_result_list == [9]:
            value1 = 3
        else:
            value1 = 0

        # Player2
        player2 = faceCheckings()
        result = duals()
        if result and 'rps_result' in result:
            rps_result_list = result['rps_result']
            if rps_result_list == [0]:
                value2 = 1
            elif rps_result_list == [5]:
                value2 = 2
            elif rps_result_list == [9]:
                value2 = 3
            else:
                value2 = 0

        # if rps_result[0]['rps'] == 'rock':
        #     if rps_result[1]['rps'] == 'rock':
        #         text = 'Tie'
        #     elif rps_result[1]['rps'] == 'paper':
        #         text = 'Paper wins'
        #         winner = 1
        #     elif rps_result[1]['rps'] == 'scissors':
        #         text = 'Rock wins'
        #         winner = 0
        # elif rps_result[0]['rps'] == 'paper':
        #     if rps_result[1]['rps'] == 'rock':
        #         text = 'Paper wins'
        #         winner = 0
        #     elif rps_result[1]['rps'] == 'paper':
        #         text = 'Tie'
        #     elif rps_result[1]['rps'] == 'scissors':
        #         text = 'Scissors wins'
        #         winner = 1
        # elif rps_result[0]['rps'] == 'scissors':
        #     if rps_result[1]['rps'] == 'rock':
        #         text = 'Rock wins'
        #         winner = 1
        #     elif rps_result[1]['rps'] == 'paper':
        #         text = 'Scissors wins'
        #         winner = 0
        #     elif rps_result[1]['rps'] == 'scissors':
        #         text = 'Tie'

        # if winner is not None:
        #     cv2.putText(img, text=text, org=(int(
        #         img.shape[1] / 5), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
        # cv2.putText(img, text="Return enter r", org=(int(
        #     img.shape[1] / 5), 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)

        # Compare values and show winner
        if value1 is not None and value2 is not None:
            if value1 > value2:
                winner_image = resize_image(player1)
                winner_text = f"Player 1 Wins! (Value: {value1})"
                cv2.putText(winner_image, text="Player1 win", org=(
                    250, 330), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=(0, 255, 0), thickness=3)

            elif value2 > value1:
                winner_image = resize_image(player2)
                winner_text = f"Player 2 Wins! (Value: {value2})"
                cv2.putText(winner_image, text="Player2 win", org=(
                    250, 330), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=(0, 255, 0), thickness=3)
            else:
                winner_image = resize_image(player1)
                winner_text = "It's a Tie!"
                cv2.putText(winner_image, text="Tie one more Play", org=(
                    250, 330), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=(0, 255, 0), thickness=3)

            cv2.imshow("Winner", winner_image)
            # print(winner_text)

        key = cv2.waitKey(0)
        if key == ord('r'):
            cv2.destroyAllWindows()
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

cv2.destroyAllWindows()