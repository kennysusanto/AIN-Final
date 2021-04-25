from multiprocessing import Process, Pipe


def f(conn, name):
    conn.send(f'a string from {name}')
    print(conn.recv())
    conn.close()


if __name__ == "__main__":
    a_conn, b_conn = Pipe()

    a = Process(target=f, args=(a_conn, 'a',))

    b = Process(target=f, args=(b_conn, 'b',))

    a.start()
    b.start()

    # print(b_conn.recv())
    # print(a_conn.recv())

    a.join()
    b.join()
