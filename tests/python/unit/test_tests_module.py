import astra


def test_cuda(capsys):
    astra.test_CUDA()
    n_stdout_lines = len(capsys.readouterr().out.split('\n')) - 1
    assert n_stdout_lines == 5


def test_no_cuda(capsys):
    astra.test_noCUDA()
    n_stdout_lines = len(capsys.readouterr().out.split('\n')) - 1
    assert n_stdout_lines == 2
