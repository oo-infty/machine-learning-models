{
  description = "ML Environment in NixOS";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311Packages.python
            python311Packages.torch-bin
            python311Packages.torchvision-bin
            python311Packages.numpy
            python311Packages.pandas
            python311Packages.matplotlib
            python311Packages.pillow
            python311Packages.opencv4
            python311Packages.tqdm
            python311Packages.albumentations
            python311Packages.black
            nodePackages.pyright
            ruff
            ruff-lsp
          ];
        };
      });
}

