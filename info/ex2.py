MIN = 1
MAX = 100
greater = '>'
equal = '='
less = '<'
print(f"Think of a number between {MIN} and {MAX} !")
count = 0
max = MAX
min = MIN
while True:
    try:
        print(f"Is Your number greater ({greater}), eaual ({equal}), or less ({less}) than {int((max + min)/2)}?")
        user_input = str(input("Please answer <,=, or >!"))
        if user_input is greater:
            min = int((max + min)/2) + 1
            if min > max:
                print("You are cheatting!")
                break
            count = count + 1
        elif user_input is less:
            max = int((max + min) / 2) - 1
            if min > max:
                print("You are cheatting!")
                break
            count = count + 1
        elif user_input is equal:
            guess = int((max + min) / 2)
            count = count + 1
            print("I have guessed it!")
            print(f"I needed {count} steps!")
            break
        else:
            print("Please answer <,=, or >!")
            continue
    except ValueError:
        print("Please answer <,=, or >!")
