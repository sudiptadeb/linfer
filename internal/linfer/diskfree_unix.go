//go:build unix

package linfer

import "syscall"

// diskFree is the free bytes on the filesystem holding path (or the nearest
// existing parent, when path is not there yet). Statfs's field types differ
// between darwin and linux, hence the conversions.
func diskFree(path string) (uint64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(nearestExisting(path), &st); err != nil {
		return 0, err
	}
	return uint64(st.Bavail) * uint64(st.Bsize), nil //nolint:unconvert // Bsize is int64 on linux, uint32 on darwin
}
