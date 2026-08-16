#include "memory_pool_cpu.h"

void Forge::MemoryPoolCPU::RadixTree::destroy_node(RadixNode* node, int level) {
    if (!node) return;
    if (level < 2) {
        for (void* p : node->entries) {
            if (p) destroy_node(static_cast<RadixNode*>(p), level + 1);
        }
    }
    delete node;
}

void* Forge::MemoryPoolCPU::allocate(std::size_t size) {
    if (size<Bin::ALIGNMENT) size = Bin::ALIGNMENT;
    else size = std::bit_ceil(size);
    constexpr auto chunk_size {Bin::CHUNK_SIZE};

    const auto bin_idx {std::countr_zero(size)};
    auto& bin {m_bins[bin_idx]};
    if (bin) {
        if (auto* block {bin->allocate()}; block) return block;
        const auto chunks_to_allocate {std::max<std::size_t>(1, bin->m_chunks_allocated)};
        bin->extendBlocks(chunks_to_allocate);
        auto* mem {bin->m_mem_allocated.back()};
        for (std::size_t chunk{}; chunk<chunks_to_allocate; ++chunk) m_radix_tree.radix_insert(mem+chunk*chunk_size, bin.get());
        return bin->allocate();
    }
    const auto chunks_to_allocate {(size+chunk_size-1)/chunk_size};
    bin = std::make_unique<Bin>(size, chunks_to_allocate);
    auto* mem {bin->m_mem_allocated.back()};
    for (std::size_t chunk{}; chunk<chunks_to_allocate; ++chunk) m_radix_tree.radix_insert(mem+chunk*chunk_size, bin.get());
    return bin->allocate();
}

void Forge::MemoryPoolCPU::RadixTree::radix_insert(void *ptr, Bin* bin_ptr) {

    const auto addr {reinterpret_cast<std::uintptr_t>(ptr)};
    constexpr std::size_t OFFSET {std::countr_zero(Bin::CHUNK_SIZE)};
    constexpr auto indexing_bits{Forge::MemoryPoolCPU::INDEXING_BITS};
    constexpr auto MASK {(1ULL<<indexing_bits)-1};

    const auto idx {addr>>OFFSET};
    const auto L1 {(idx >> (indexing_bits*2)) & MASK};
    const auto L2 {(idx >> indexing_bits) & MASK};
    const auto L3 {idx & MASK};

    assert(L1 < (1ULL << indexing_bits));
    assert(L2 < (1ULL << indexing_bits));
    assert(L3 < (1ULL << indexing_bits));

    if (!m_radix_root) m_radix_root = new RadixNode();
    if (!(m_radix_root->entries[L1]))  m_radix_root->entries[L1] = new RadixNode();
    auto* N2 {static_cast<RadixNode*>(m_radix_root->entries[L1])};
    if (!(N2->entries[L2])) N2->entries[L2] = new RadixNode();
    auto* N3 {static_cast<RadixNode*>(N2->entries[L2])};

    N3->entries[L3] = bin_ptr;
}

Forge::MemoryPoolCPU::Bin* Forge::MemoryPoolCPU::RadixTree::radix_lookup(void *ptr) const {
    const auto addr {reinterpret_cast<std::uintptr_t>(ptr)};
    constexpr auto indexing_bits {INDEXING_BITS};
    constexpr std::size_t OFFSET {std::countr_zero(Bin::CHUNK_SIZE)};
    constexpr auto MASK {(1ULL<<indexing_bits)-1};

    const auto idx {addr>>OFFSET};
    const auto L1 {(idx >> (indexing_bits*2)) & MASK};
    const auto L2 {(idx >> indexing_bits) & MASK};
    const auto L3 {idx & MASK};

    assert(L1 < (1ULL << indexing_bits));
    assert(L2 < (1ULL << indexing_bits));
    assert(L3 < (1ULL << indexing_bits));

    if (!m_radix_root) return nullptr;
    RadixNode *N2{}, *N3{};
    N2 = static_cast<RadixNode*>(m_radix_root->entries[L1]);
    if (!N2) return nullptr;
    N3 = static_cast<RadixNode*>(N2->entries[L2]);
    if (!N3) return nullptr;
    auto* bin_ptr {static_cast<Bin*>(N3->entries[L3])};
    return bin_ptr?bin_ptr:nullptr;
}

void Forge::MemoryPoolCPU::free(void* ptr) {
    if (!ptr) return;
    if (auto* bin_ptr {m_radix_tree.radix_lookup(ptr)}; bin_ptr) bin_ptr->free(ptr);
}

void Forge::MemoryPoolCPU::hard_clear_cache() {
    m_radix_tree.clear();
    for (std::size_t i {}; i<std::size(m_bins); ++i){if (m_bins[i]) m_bins[i]->free_cache();}
}

Forge::MemoryPoolCPU::Bin::Bin(std::size_t class_size, std::size_t chunks) {
    m_class_size = std::bit_ceil(class_size);
    if (m_class_size>(chunks*CHUNK_SIZE)) throw std::invalid_argument(
        std::format("number of chunks are less to at least initialize one block of {}B bin size", class_size)
    );
    extendBlocks(chunks);
}

void Forge::MemoryPoolCPU::Bin::extendBlocks(std::size_t chunks) {
    auto* mem {static_cast<byte*>(ALIGNED_ALLOC(chunks*CHUNK_SIZE, CHUNK_SIZE))};
    if (!mem) throw std::bad_alloc();
    m_mem_allocated.push_back(mem);
    const std::size_t blocks {(chunks*CHUNK_SIZE)/m_class_size};
    for (std::size_t block{}; block<blocks; ++block) {
        auto* block_base {mem+block*m_class_size};
        auto* block_ptr {reinterpret_cast<freeBlock*>(block_base)};
        block_ptr->next_block = m_freeListHead;
        m_freeListHead = block_ptr;
    }
    m_totalBlocks += blocks; m_freeBlocks += blocks, m_chunks_allocated += chunks;
}

void* Forge::MemoryPoolCPU::Bin::allocate() {
    // std::cout<<"Allocating From Bin Size "<<m_class_size<<"\n";
    if (!m_freeListHead) [[unlikely]] return nullptr;
    auto* free_block {m_freeListHead};
    PREFETCH(free_block->next_block, 0, 1);
    m_freeListHead = free_block->next_block;
    m_freeBlocks--;
    return free_block;
}

void Forge::MemoryPoolCPU::Bin::free(void* ptr) {
    // std::cout<<"Deallocating To Bin Size "<<m_class_size<<"\n";
    auto* freed_block {static_cast<freeBlock*>(ptr)};
    freed_block->next_block = m_freeListHead;
    m_freeListHead = freed_block;
    m_freeBlocks++;
}

void Forge::MemoryPoolCPU::Bin::free_cache() {
    for (const auto& mem: m_mem_allocated) ALIGNED_FREE(mem);
    m_mem_allocated.clear();
    m_totalBlocks = m_freeBlocks = 0;
}


Forge::MemoryPoolCPU::Bin::~Bin() {
    free_cache();
}