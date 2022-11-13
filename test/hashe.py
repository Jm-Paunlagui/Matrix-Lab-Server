from hashlib import md5


def hashd(string):
    return md5(string.encode()).hexdigest()


print(hashd("test"))
print(hashd("test"))
