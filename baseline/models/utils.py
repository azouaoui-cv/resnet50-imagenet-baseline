def print_info_net(net):
    total_params = sum(
        [x.numel() for x in filter(lambda p: p.requires_grad, net.parameters())]
    )
    total_layers = len(
        list(
            filter(
                lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()
            )
        )
    )
    print(f"Total nb params {total_params:,}")
    print(f"Total nb layers {total_layers}")
