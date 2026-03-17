def main():
    from tests.test_encoding import test_shapes
    from tests.test_signal_generator import test_generators
    test_shapes()
    test_generators()
    print("OK")

if __name__ == "__main__":
    main()
